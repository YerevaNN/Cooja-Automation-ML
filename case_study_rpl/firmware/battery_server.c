/*
* Battery-aware RPL Server (Root Node)
* 
* This is the root/sink node that:
* - Establishes itself as the DODAG root
* - Receives messages from all client nodes
* - Logs message receipt with originator and immediate neighbor info
* - Has unlimited battery (never depletes)
*/

#include "contiki.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "net/ipv6/simple-udp.h"
#include "net/ipv6/uip-sr.h"
#include "sys/energest.h"
 
 // Allow overriding expected nodes via build define
#ifndef EXPECTED_NODES
#define EXPECTED_NODES -1
#endif

#include "sys/node-id.h"

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

/* Log configuration */
#include "sys/log.h"
#define LOG_MODULE "Server"
#define LOG_LEVEL LOG_LEVEL_INFO

/* UDP configuration */
#define UDP_SERVER_PORT 5678
#define UDP_CLIENT_PORT 8765

/* Battery configuration */
#define UNLIMITED_BATTERY 0xFFFFFFFF
#define BATTERY_CHECK_INTERVAL (15 * CLOCK_SECOND)

/* Message structure */
typedef struct {
    uint16_t originator_id;
    uint32_t sequence;
} message_t;

 
 /* Global state */
 static struct simple_udp_connection udp_conn;
 static uint32_t messages_received = 0;
 
 /*---------------------------------------------------------------------------*/
 /* Helper Functions */
 /*---------------------------------------------------------------------------*/
 
 static unsigned long
 ticks_to_seconds(uint64_t ticks)
 {
     return (unsigned long)(ticks / ENERGEST_SECOND);
 }
 
 static void
 log_energest_stats(void)
 {
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
 
     // Root has unlimited battery
     LOG_INFO("BATTERY: initial=%lumAh consumed=0mAh remaining=%lumAh status=alive uptime=%lus\n",
              (unsigned long)UNLIMITED_BATTERY, (unsigned long)UNLIMITED_BATTERY, clock_seconds());
 }
 
 static uint16_t
 get_neighbor_id_from_addr(const uip_ipaddr_t *addr)
 {
     // Extract node ID from the IPv6 address
     // In Cooja, the node ID is typically in the last byte of the address
     if(addr != NULL) {
         uint16_t id = addr->u8[15];
         if(id == 0) {
             // Sometimes it's in a different position, check byte 14
             id = addr->u8[14];
         }
         return id;
     }
     return 0;
 }
 
 /*---------------------------------------------------------------------------*/
 /* UDP Callback */
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
     /* IMPORTANT: Validate message size before accessing data
      * This prevents buffer overruns if we receive malformed or corrupted packets
      * In a real network, packets can be truncated or corrupted */
     if(datalen >= sizeof(message_t)) {
         message_t msg;
         memcpy(&msg, data, sizeof(message_t));
         
         /* Extract neighbor ID from the IPv6 address of immediate sender
          * This tells us which node directly forwarded this message to us */
         uint16_t neighbor_id = get_neighbor_id_from_addr(sender_addr);
         
         messages_received++;
         
         /* Log format: [my_id] Received msg seq=X from originator=Y via neighbor=Z
          * - my_id: This node (the root)
          * - originator: The client that created the message
          * - neighbor: The node that directly sent it to us (could be originator or a forwarder) */
         LOG_INFO("[%u] Received msg seq=%"PRIu32" from originator=%u via neighbor=%u (total_received=%"PRIu32")\n",
                  node_id, msg.sequence, msg.originator_id, neighbor_id, messages_received);
         
         // Also log the full IPv6 address for debugging
         LOG_INFO("  Full neighbor address: ");
         LOG_INFO_6ADDR(sender_addr);
         LOG_INFO_("\n");
     } else {
         LOG_WARN("Received malformed message of size %u\n", datalen);
     }
 }
 
 /*---------------------------------------------------------------------------*/
 PROCESS(server_process, "Battery Server Process");
 AUTOSTART_PROCESSES(&server_process);
 /*---------------------------------------------------------------------------*/
 PROCESS_THREAD(server_process, ev, data)
 {
     static struct etimer periodic_timer;
     static struct etimer battery_timer;
     static bool network_formed = false;
     
     PROCESS_BEGIN();
     
     LOG_INFO("Starting server node (ID: %u)\n", node_id);
     
     /* Initialize as DAG root */
     NETSTACK_ROUTING.root_start();
     
     /* Register UDP connection */
     simple_udp_register(&udp_conn, UDP_SERVER_PORT, NULL,
                        UDP_CLIENT_PORT, udp_rx_callback);
     
     /* Wait for network to form - give nodes time to join */
     etimer_set(&periodic_timer, CLOCK_SECOND * 3);
     unsigned int nodes_joined = 0;
     // initialize to satisfy MSP430 -Werror=uninitialized
     static unsigned int expected_nodes = 0;

     if (EXPECTED_NODES > 0) {
         expected_nodes = (unsigned int)EXPECTED_NODES;
         LOG_INFO("Waiting for %u nodes to join the network...\n", expected_nodes);
     } else {
         LOG_INFO("Waiting up to 60s for network formation (no expected count)\n");
     }
     
     /* Keep checking until all nodes join or 60 seconds elapsed */
     while(!network_formed && clock_seconds() < 60) {
         PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&periodic_timer));
         
         nodes_joined = uip_sr_num_nodes(); //TODO: why this is returning 0 for the server in certain runs?
         
        //  LOG_INFO("Network formation: %u/%u nodes joined (T=%lus)\n", 
        //          nodes_joined, expected_nodes, clock_seconds());
         

         if (EXPECTED_NODES > 0) {
           LOG_INFO("Network formation: %u/%u nodes joined (T=%lus)\n", 
                   nodes_joined, expected_nodes, clock_seconds());
         } else {
           LOG_INFO("Network formation: %u nodes joined (T=%lus)\n", 
                   nodes_joined, clock_seconds());
         }

         if(EXPECTED_NODES > 0 && nodes_joined >= expected_nodes) {
             network_formed = true;
             LOG_INFO("SUCCESS: All nodes joined at T=%lus! Network ready.\n", 
                     clock_seconds());
         } else {
             etimer_reset(&periodic_timer);
         }
     }
     
     if(!network_formed) {
         LOG_WARN("WARNING: Only %u/%u nodes joined after 60s. Proceeding anyway.\n",
                 nodes_joined, expected_nodes);
         network_formed = true;  // Proceed anyway
     }
     
     LOG_INFO("=== NETWORK OPERATIONAL at T=%lus ===\n", clock_seconds());
     
     /* Set up battery logging timer */
     etimer_set(&battery_timer, BATTERY_CHECK_INTERVAL);
     
     /* Main loop - just wait for messages and log periodically */
     while(1) {
         PROCESS_WAIT_EVENT();
         
         if(ev == PROCESS_EVENT_TIMER && data == &battery_timer) {
             log_energest_stats();
             LOG_INFO("Messages received so far: %"PRIu32"\n", messages_received);
             etimer_reset(&battery_timer);
         }
     }
     
     PROCESS_END();
 }
 /*---------------------------------------------------------------------------*/
