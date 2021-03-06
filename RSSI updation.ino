#include "ThingSpeak.h"
#include <ESP8266WiFi.h>
// Use this file to store all of the private credentials 
// and connection details

#define SECRET_SSID "Hotspot"    // replace MySSID with your WiFi network name
#define SECRET_PASS "987654321"  // replace MyPassword with your WiFi password

#define SECRET_CH_ID 949069      // replace 0000000 with your channel number
#define SECRET_WRITE_APIKEY "XR7FUXM1BX7AX18T"   // replace XYZ with your channel write API Key


const char* ssid = "Hotspot";   // your network SSID (name) 
const char* pass = "987654321";   // your network password
WiFiClient  client;

unsigned long myChannelNumber = 949069;
const char * myWriteAPIKey = "XR7FUXM1BX7AX18T";
int Sensor= A0;
int Val=0;
int a=0;


void setup() {
  Serial.begin(115200);  // Initialize serial
  WiFi.mode(WIFI_STA); 
  ThingSpeak.begin(client);  // Initialize ThingSpeak
  pinMode(A0,INPUT);
 
}

void loop() {
  Val=analogRead(Sensor);
  Serial.println(Val);
  a=WiFi.RSSI();
  Serial.println(a);
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
   delay(10000);
  int x = ThingSpeak.writeField(myChannelNumber,5,Val, myWriteAPIKey);
  if((x == 200)){
    Serial.println("Channel update successful.");
  }
  delay(10000); // Wait 20 seconds to update the channel again
  int y = ThingSpeak.writeField(myChannelNumber,6,a, myWriteAPIKey);
  if((y == 200)){
    Serial.println("Channel update successful.");
  }
}
