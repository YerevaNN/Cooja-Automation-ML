#ifndef PROJECT_CONF_H_
#define PROJECT_CONF_H_

/* Energest: keep it ON so CPU/LPM/Radio accounting works */
#ifndef ENERGEST_CONF_ON
#define ENERGEST_CONF_ON 1
#endif

/* Keep ROM usage low on Sky  (But what about z1) */ 
// #define UIP_CONF_TCP 0

/* RPL mode: storing reduces ROM on the root (no uip-sr) */
// #define RPL_CONF_WITH_STORING 1
#undef RPL_CONF_WITH_STORING
#define RPL_CONF_WITH_STORING 0

#define UIP_SR_CONF_LINK_NUM  75


// // /* Table sizes sized for ~25 clients */
// #undef  NETSTACK_MAX_ROUTE_ENTRIES
// #define NETSTACK_MAX_ROUTE_ENTRIES 32
// #undef  NBR_TABLE_CONF_MAX_NEIGHBORS
// #define NBR_TABLE_CONF_MAX_NEIGHBORS 24

#define NETSTACK_CONF_RDC contikimac_driver
#undef CONTIKIMAC_CONF_CHANNEL_CHECK_RATE
#define CONTIKIMAC_CONF_CHANNEL_CHECK_RATE 8 // TODO: does it even work right? does changing this 16 help? 

// /* Network stack configuration */
// #define NETSTACK_MAX_ROUTE_ENTRIES 25
// #define NBR_TABLE_CONF_MAX_NEIGHBORS 10

/* PAN ID (keep your existing value) */
#define IEEE802154_CONF_PANID 0xABCD

/* Logging: keep app logs, cut the rest to fit ROM */
#define LOG_CONF_WITH_COMPACT_ADDR 1
// #define LOG_CONF_LEVEL_RPL      LOG_LEVEL_NONE
#define LOG_CONF_LEVEL_IPV6     LOG_LEVEL_NONE
#define LOG_CONF_LEVEL_TCPIP    LOG_LEVEL_NONE
#define LOG_CONF_LEVEL_6LOWPAN  LOG_LEVEL_NONE
// #define LOG_CONF_LEVEL_MAC      LOG_LEVEL_NONE
#define LOG_CONF_LEVEL_COAP     LOG_LEVEL_NONE
#define LOG_CONF_LEVEL_MAIN     LOG_LEVEL_INFO  // TODO: SET BACK TO NONE LATER
#define LOG_CONF_LEVEL_APP      LOG_LEVEL_INFO  /* raise/lower as needed */

#define LOG_CONF_LEVEL_RPL LOG_LEVEL_INFO

#define LOG_CONF_WITH_COMPACT_ADDR 1

#ifndef UIP_CONF_STATISTICS
#define UIP_CONF_STATISTICS 1
#endif


#endif