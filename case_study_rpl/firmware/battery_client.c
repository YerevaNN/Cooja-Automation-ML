/*
* Battery-aware RPL Client
* 
* This is a client node that:
* - Joins the RPL network
* - Periodically sends messages to the root
* - Tracks battery consumption and shuts down when depleted
* - Logs all activities for analysis
*/

#include "contiki.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "net/ipv6/simple-udp.h"
#include "sys/energest.h"
#include "sys/node-id.h"
#include "random.h"

// recording ip stats
#include "net/ipv6/uip.h"
#include <stdio.h> 
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>

/* Log configuration */
#include "sys/log.h"
#define LOG_MODULE "Client"
#define LOG_LEVEL LOG_LEVEL_INFO

/* UDP configuration */
#define UDP_CLIENT_PORT 8765
#define UDP_SERVER_PORT 5678
 
/* Timing configuration */
// 5-second interval is quite aggressive but reasonable for testing:
#define SEND_INTERVAL (5 * CLOCK_SECOND)   // Aggressive - good for quick tests
// #define SEND_INTERVAL (10 * CLOCK_SECOND)  // Current - balanced
// #define SEND_INTERVAL (30 * CLOCK_SECOND)  // Conservative - longer battery life
// Uncomment below for random interval (uniform distribution)
// #define SEND_INTERVAL_MIN (5 * CLOCK_SECOND)
// #define SEND_INTERVAL_MAX (15 * CLOCK_SECOND)

#define BATTERY_CHECK_INTERVAL (10 * CLOCK_SECOND) // this is where we get uptime and other metrics
#define STARTUP_DELAY (5 * CLOCK_SECOND)  // Initial delay before checking reachability

// Battery configuration
#define MIN_BATTERY_MAH 25    // Dies in ~5-10 minutes with multiplier
#define MAX_BATTERY_MAH 100    // Survives X+ miuntes

// Current consumption - modest multiplier for faster simulation
#define MULTIPLIER 30  // Instead of 50

/* Platform-specific current consumption in mA (micro-amperes) from Sky Mote datasheet */
/* Current consumption in mA */
#define CPU_CURRENT_MA 1.8 * MULTIPLIER // Current when the CPU is running
#define LPM_CURRENT_MA 0.0545 * MULTIPLIER // Current when the CPU is in low power mode
#define DEEP_LPM_CURRENT_MA 0.0135 * MULTIPLIER // Current when the CPU is in deep low power mode
#define RADIO_LISTEN_CURRENT_MA 20.0 * MULTIPLIER // Current when the radio is in listen mode
#define RADIO_TRANSMIT_CURRENT_MA 17.4 * MULTIPLIER // Current when the radio is in transmit mode
 
/* Message structure */
typedef struct {
    uint16_t originator_id;
    uint32_t sequence;
} message_t; 


#ifndef EXPECTED_NODES
#define EXPECTED_NODES -1
#endif
 
static unsigned long
ticks_to_seconds(uint64_t ticks)
{
    return (unsigned long)(ticks / ENERGEST_SECOND);
}
 
 /* Global state */
 static struct simple_udp_connection udp_conn;
 static uint32_t initial_battery_mah;
 static bool node_shutdown = false;
 static uint32_t msg_sequence = 0;
 static uint32_t messages_sent = 0;
 static uint32_t messages_failed = 0;
 
 /*---------------------------------------------------------------------------*/
 /* Helper Functions */
 /*---------------------------------------------------------------------------*/
 
 static uint32_t
 calculate_energy_consumption_mah(void)
 {
     energest_flush();
     
     uint64_t cpu_time = energest_type_time(ENERGEST_TYPE_CPU);
     uint64_t lpm_time = energest_type_time(ENERGEST_TYPE_LPM);
     uint64_t transmit_time = energest_type_time(ENERGEST_TYPE_TRANSMIT);
     uint64_t listen_time = energest_type_time(ENERGEST_TYPE_LISTEN);
     
     double total_mah = (ticks_to_seconds(cpu_time) * CPU_CURRENT_MA +
                        ticks_to_seconds(lpm_time) * LPM_CURRENT_MA +
                        ticks_to_seconds(listen_time) * RADIO_LISTEN_CURRENT_MA +
                        ticks_to_seconds(transmit_time) * RADIO_TRANSMIT_CURRENT_MA) / 3600.0;
     
     return (uint32_t)(total_mah * 100);  // Return in units of 0.01 mAh for precision
 }
 
 static void
 log_energest_and_battery(void)
 {

    LOG_INFO("IPSTATS: sent=%lu recv=%lu fwd=%lu drop=%lu\n",
        (unsigned long)uip_stat.ip.sent,
        (unsigned long)uip_stat.ip.recv,
        (unsigned long)uip_stat.ip.forwarded,
        (unsigned long)uip_stat.ip.drop);


     energest_flush();
     
     unsigned long cpu = ticks_to_seconds(energest_type_time(ENERGEST_TYPE_CPU));
     unsigned long lpm = ticks_to_seconds(energest_type_time(ENERGEST_TYPE_LPM));
     unsigned long deep_lpm = ticks_to_seconds(energest_type_time(ENERGEST_TYPE_DEEP_LPM));
     unsigned long listen = ticks_to_seconds(energest_type_time(ENERGEST_TYPE_LISTEN));
     unsigned long transmit = ticks_to_seconds(energest_type_time(ENERGEST_TYPE_TRANSMIT));
     unsigned long total = ticks_to_seconds(ENERGEST_GET_TOTAL_TIME());
     unsigned long off = total > (listen + transmit) ? total - listen - transmit : 0;
     
     LOG_INFO("ENERGEST: CPU=%lus LPM=%lus DEEP_LPM=%lus LISTEN=%lus TRANSMIT=%lus OFF=%lus TOTAL=%lus\n",
              cpu, lpm, deep_lpm, listen, transmit, off, total);
     
     uint32_t consumed = calculate_energy_consumption_mah();
     uint32_t consumed_mah = consumed / 100;  // Convert back to mAh for display
     uint32_t remaining = initial_battery_mah > consumed_mah ? initial_battery_mah - consumed_mah : 0;
     const char *status = node_shutdown ? "dead" : "alive";
     unsigned long uptime = clock_seconds();
     
     LOG_INFO("BATTERY: initial=%lumAh consumed=%lumAh remaining=%lumAh status=%s uptime=%lus\n",
              (unsigned long)initial_battery_mah, (unsigned long)consumed_mah, 
              (unsigned long)remaining, status, uptime);
 }
 
 static void
 init_battery_system(void)
 {
     // Sample from uniform distribution [MIN_BATTERY_MAH, MAX_BATTERY_MAH]
     initial_battery_mah = MIN_BATTERY_MAH + 
                          (random_rand() % (MAX_BATTERY_MAH - MIN_BATTERY_MAH + 1));
     LOG_INFO("Node %u battery: %lu mAh\n", node_id, (unsigned long)initial_battery_mah);
 }
 
 static bool
 check_battery_status(void)
 {
     if(node_shutdown) {
         return false;
     }
     
     uint32_t energy_consumed = calculate_energy_consumption_mah() / 100;  // Convert to mAh
     
     if(energy_consumed >= initial_battery_mah) {
         LOG_INFO("Battery depleted! Shutting down. Energy used: %lu mAh\n", 
                 (unsigned long)energy_consumed);
         LOG_INFO("Final stats: sent=%"PRIu32" failed=%"PRIu32"\n", 
                 messages_sent, messages_failed);
         
         /* Turn off the radio */
         NETSTACK_MAC.off();
         node_shutdown = true;
         return false;
     }
     
     LOG_INFO("Energy consumed: %lu/%lu mAh\n", 
             (unsigned long)energy_consumed, (unsigned long)initial_battery_mah);
     return true;
 }
 
 static clock_time_t
 get_next_send_interval(void)
 {
     // Fixed interval version
     return SEND_INTERVAL;
     
     // Uncomment for random interval version
     // return SEND_INTERVAL_MIN + 
     //        (random_rand() % (SEND_INTERVAL_MAX - SEND_INTERVAL_MIN));
 }
 
 /*---------------------------------------------------------------------------*/
 /* UDP Callback - Only called if this node is the destination */
 /*---------------------------------------------------------------------------*/
 static void
 udp_rx_callback(struct simple_udp_connection *c,
                 const uip_ipaddr_t *sender_addr,
                 uint16_t sender_port,
                 const uip_ipaddr_t *receiver_addr,
                 uint16_t receiver_port,
                 const uint8_t *data,
                 uint16_t datalen)
 {
     /* This callback is ONLY triggered when:
      * 1. A UDP packet arrives at OUR port (UDP_CLIENT_PORT)
      * 2. We are the final destination (not just forwarding)
      * 
      * RPL forwarding happens at a lower layer and doesn't trigger this!
      * In our architecture, clients don't expect any messages, so this
      * would only fire if there's unexpected traffic or a misconfiguration */
     
     LOG_WARN("Unexpected message received at client from ");
     LOG_WARN_6ADDR(sender_addr);
     LOG_WARN_(" - clients should not receive messages in this architecture!\n");
 }
 
 /*---------------------------------------------------------------------------*/
 PROCESS(client_process, "Battery Client Process");
 AUTOSTART_PROCESSES(&client_process);
 /*---------------------------------------------------------------------------*/
 PROCESS_THREAD(client_process, ev, data)
 {
     static struct etimer send_timer;
     static struct etimer battery_timer;
     static struct etimer network_wait_timer;
     static uip_ipaddr_t root_addr;
     static bool sending_started = false;
     
     PROCESS_BEGIN();
     
     LOG_INFO("Starting client node (ID: %u)\n", node_id);
     random_init(node_id);
     /* Initialize battery */
     init_battery_system();
     
     /* Register UDP connection */
     simple_udp_register(&udp_conn, UDP_CLIENT_PORT, NULL,
                        UDP_SERVER_PORT, udp_rx_callback);
     
     /* Wait for network reachability */
     etimer_set(&send_timer, STARTUP_DELAY);
    
     
     /* Wait for network formation to complete */
     PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&network_wait_timer));
     LOG_INFO("Starting periodic message transmission\n");
     
     /* Set up timers */
     etimer_set(&send_timer, get_next_send_interval());
     etimer_set(&battery_timer, BATTERY_CHECK_INTERVAL);
     sending_started = true;
     
     /* Main loop */
     while(1) {
         /* Wait for any event to occur */
         PROCESS_WAIT_EVENT();
         
         /* The 'ev' parameter tells us what type of event occurred
          * The 'data' parameter provides event-specific information */
         
         if(ev == PROCESS_EVENT_TIMER) {
             /* A timer expired - 'data' points to which timer */
             
             if(data == &send_timer && !node_shutdown && sending_started) {
                 /* The send timer expired - time to send a message */
                 
                 if(NETSTACK_ROUTING.node_is_reachable() &&
                 NETSTACK_ROUTING.get_root_ipaddr(&root_addr)) {
                     message_t msg;
                     msg.originator_id = node_id;
                     msg.sequence = msg_sequence++;
                     
                     LOG_INFO("[%u] Sending msg seq=%"PRIu32" to root\n", 
                             node_id, msg.sequence);
                     
                     simple_udp_sendto(&udp_conn, &msg, sizeof(msg), &root_addr);
                     messages_sent++;
                     
                     /* Log statistics every 10 messages for monitoring */
                     if(messages_sent % 10 == 0) {
                         LOG_INFO("Stats: sent=%"PRIu32" failed=%"PRIu32"\n", 
                                 messages_sent, messages_failed);
                     }
                 } else {
                     LOG_WARN("Not reachable, skipping send (seq=%"PRIu32")\n", 
                             msg_sequence);
                     messages_failed++;
                     msg_sequence++;  // Still increment to track missed sends
                 }
                 
                 etimer_set(&send_timer, get_next_send_interval());
                 
             } else if(data == &battery_timer) {
                 /* The battery check timer expired - time to check energy */
                 
                 log_energest_and_battery();
                 
                 if(!check_battery_status()) {
                     log_energest_and_battery();  // Log one last time
                     LOG_INFO("Node %u shutting down due to battery depletion\n", node_id);
                     PROCESS_EXIT();  // Terminate this process
                 }
                 
                 etimer_reset(&battery_timer);
             }
             /* If neither timer, ignore (shouldn't happen) */
         }
         /* Other event types could be handled here if needed */
     }
     
     PROCESS_END();
 }
 /*---------------------------------------------------------------------------*/
