#include "ThingSpeak.h"
#include <ESP8266WiFi.h>
// Use this file to store all of the private credentials 
// and connection details

#define SECRET_SSID ""    // replace MySSID with your WiFi network name
#define SECRET_PASS ""  // replace MyPassword with your WiFi password

#define SECRET_CH_ID 948356      // replace 0000000 with your channel number
#define SECRET_WRITE_APIKEY "M8WQX2HQUF7817N1"   // replace XYZ with your channel write API Key


const char* ssid = "";   // your network SSID (name) 
const char* pass = "";   // your network password
WiFiClient  client;

unsigned long myChannelNumber = 930711;
const char * myWriteAPIKey = "V42QVLVG4JKGF6H5";
int Sensor= A0;
int Val=0;


void setup() {
  Serial.begin(115200);  // Initialize serial
  WiFi.mode(WIFI_STA); 
  ThingSpeak.begin(client);  // Initialize ThingSpeak
  pinMode(A0,INPUT);
 
}

void loop() {
  Val=analogRead(Sensor);
  Serial.println(Val);
  Serial.println(WiFi.RSSI());
  // Connect or reconnect to WiFi
  if(WiFi.status() != WL_CONNECTED){
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(SECRET_SSID);
    while(WiFi.status() != WL_CONNECTED){
      WiFi.begin(ssid, pass);  // Connect to WPA/WPA2 network. Change this line if using open or WEP network
      Serial.print(".");
      delay(5000);     
    } 
    Serial.println("\nConnected.");
    
    Serial.println("Sensor Function starts"); //update the type of sensor
    
  }
  
  // Write to ThingSpeak. There are up to 8 fields in a channel, allowing you to store up to 8 different
  // pieces of information in a channel.  Here, we write to field 1.
  int x = ThingSpeak.writeField(myChannelNumber,2,Val, myWriteAPIKey);
  if((x == 200)){
    Serial.println("Channel update successful.");
  }
  delay(10000); // Wait 20 seconds to update the channel again
}
