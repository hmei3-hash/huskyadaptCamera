#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "pi-hotspot";
const char* password = "12345678";

WebServer server(80);

const int trigPin = 18;
const int echoPin = 19;
float lastDistance = 0;

float getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH);
  return duration * 0.034 / 2.0;
}

void handleClient() {
  String json = "{\"ultrasonic_cm\":" + String(lastDistance, 2) + "}";
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected! IP: " + WiFi.localIP().toString());

  // Initialize web server once
  server.on("/reading", handleClient);
  server.begin();
  Serial.println("Server started");
}

void loop() {
  lastDistance = getDistance();
  Serial.println("Distance: " + String(lastDistance) + " cm");

  server.handleClient(); // process incoming requests

  delay(100);
}