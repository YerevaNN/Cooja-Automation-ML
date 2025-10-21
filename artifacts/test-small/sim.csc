<?xml version="1.0" encoding="UTF-8"?>
<simconf>
  <simulation>
    <title>test_small</title>
    <randomseed>123</randomseed>
    <motedelay_us>1000000</motedelay_us>
    <radiomedium>
      org.contikios.cooja.radiomediums.UDGM
      <transmitting_range>50.0</transmitting_range>
      <interference_range>100.0</interference_range>
      <success_ratio_tx>1.0</success_ratio_tx>
      <success_ratio_rx>1.0</success_ratio_rx>
    </radiomedium>
    <events>
      <logoutput>40000</logoutput>
    </events>

    <motetype>
      org.contikios.cooja.mspmote.Z1MoteType
      <identifier>z1_server</identifier>
      <description>Battery RPL Server (Root) - Z1</description>
      <source>/auto/home/aram.dovlatyan/cooja-experiments/battery_rpl_v2/src/battery_server.c</source>
      <commands>make -C /auto/home/aram.dovlatyan/cooja-experiments/battery_rpl_v2 -j$(CPUS) BUILD_DIR=/auto/home/aram.dovlatyan/cooja-experiments/battery_rpl_v2/cooja-automation-repo/runs/test_small/build -B  DEFINES+=EXPECTED_NODES=2 battery_server.z1 TARGET=z1</commands>
      <firmware>/auto/home/aram.dovlatyan/cooja-experiments/battery_rpl_v2/cooja-automation-repo/runs/test_small/build/z1/battery_server.z1</firmware>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.IPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspClock</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.Msp802154Radio</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspDefaultSerial</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspLED</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspDebugOutput</moteinterface>
    </motetype>

    <motetype>
      org.contikios.cooja.mspmote.Z1MoteType
      <identifier>z1_client</identifier>
      <description>Battery RPL Client - Z1</description>
      <source>/auto/home/aram.dovlatyan/cooja-experiments/battery_rpl_v2/src/battery_client.c</source>
      <commands>make -C /auto/home/aram.dovlatyan/cooja-experiments/battery_rpl_v2 -j$(CPUS) BUILD_DIR=/auto/home/aram.dovlatyan/cooja-experiments/battery_rpl_v2/cooja-automation-repo/runs/test_small/build -B battery_client.z1 TARGET=z1</commands>
      <firmware>/auto/home/aram.dovlatyan/cooja-experiments/battery_rpl_v2/cooja-automation-repo/runs/test_small/build/z1/battery_client.z1</firmware>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.IPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspClock</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.Msp802154Radio</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspDefaultSerial</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspLED</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspDebugOutput</moteinterface>
    </motetype>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>0.0</x>
      <y>0.0</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>1</id>
    </interface_config>
    <motetype_identifier>z1_server</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>10.0</x>
      <y>0.0</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>2</id>
    </interface_config>
    <motetype_identifier>z1_client</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>0.0</x>
      <y>10.0</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>3</id>
    </interface_config>
    <motetype_identifier>z1_client</motetype_identifier>
  </mote>
  </simulation>
  <plugin>
    org.contikios.cooja.plugins.ScriptRunner
    <plugin_config>
      <script><![CDATA[

// Headless logging for parser compatibility
TIMEOUT(60000, log.testOK());
var ROOT_ID = 1;
function formatTime(microseconds) {
  var totalMs = Math.floor(microseconds / 1000);
  var minutes = Math.floor(totalMs / 60000);
  var seconds = Math.floor((totalMs % 60000) / 1000);
  var millis = totalMs % 1000;
  var minStr = (minutes < 10 ? "0" : "") + minutes;
  var secStr = (seconds < 10 ? "0" : "") + seconds;
  var msStr = ("000" + millis).slice(-3);
  return minStr + ":" + secStr + "." + msStr;
}
while (true) {
  YIELD();
  if (msg) {
    var ts = formatTime(time);
    var module = (id == ROOT_ID) ? "Server" : "Client";
    log.log(ts + "	ID:" + id + "	[INFO: " + module + "]	" + msg + "\n");
  }
}

      ]]></script>
      <active>true</active>
    </plugin_config>
    <bounds x="0" y="0" height="100" width="100" />
  </plugin>
</simconf>