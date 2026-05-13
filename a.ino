// ---------------------------------------------------------------------------
// NewPing example: ping 1 sensor and trigger GPIO 23 HIGH for 0.5s
// when distance < 100cm (and non-zero, i.e. valid reading)
// ---------------------------------------------------------------------------

#include <NewPing.h>

#define SONAR_NUM    1      // Number of sensors
#define MAX_DISTANCE 400    // Maximum distance (in cm) to ping
#define TRIGGER_PIN  23     // Output pin to pull HIGH when object is close
#define THRESHOLD_CM 150    // Distance threshold (1 meter)
#define PULSE_MS     500    // How long to hold pin HIGH (0.5 seconds)

NewPing sonar[SONAR_NUM] = {
  NewPing(18, 5, MAX_DISTANCE),  // trig=18, echo=5
};

void setup() {
  Serial.begin(9600);
  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIGGER_PIN, LOW);
}

void loop() {
  for (uint8_t i = 0; i < SONAR_NUM; i++) {
    delay(50);  // Wait 50ms between pings (about 20 pings/sec)

    unsigned long dist = sonar[i].ping_cm();

    Serial.print(i);
    Serial.print("=");
    Serial.print(dist);
    Serial.print("cm ");

    // Trigger condition: valid reading (non-zero) AND closer than threshold
    if (dist != 0 && dist < THRESHOLD_CM) {
      Serial.print("[TRIGGERED] ");
      digitalWrite(TRIGGER_PIN, HIGH);
      delay(PULSE_MS);
      digitalWrite(TRIGGER_PIN, LOW);
    }
  }
  Serial.println();
}
