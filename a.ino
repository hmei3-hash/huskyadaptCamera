#include <NewPing.h>

#define SONAR_NUM       1
#define MAX_DISTANCE    400
#define TRIGGER_PIN     23
#define THRESHOLD_CM    100      // Social distance threshold (1m)
#define PULSE_MS        500      // How long to hold D23 HIGH

// === Consensus parameters ===
#define HISTORY_SIZE    10       // Keep last 10 readings (~1.5s window)
#define SIMILARITY_CM   10       // Two readings within 10cm count as "similar"
#define MIN_AGREEMENT   2        // Need at least 4 similar readings to trust

NewPing sonar[SONAR_NUM] = {
  NewPing(18, 5, MAX_DISTANCE),
};

// Circular buffer
unsigned long history[HISTORY_SIZE];
uint8_t historyIndex = 0;
bool historyFull = false;

void setup() {
  Serial.begin(9600);
  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIGGER_PIN, LOW);

  // Initialize buffer to 0 (invalid value marker)
  for (uint8_t i = 0; i < HISTORY_SIZE; i++) {
    history[i] = 0;
  }
}

// Count how many valid readings in buffer are "similar" to currentDist
uint8_t countSimilar(unsigned long currentDist) {
  uint8_t count = 0;
  uint8_t total = historyFull ? HISTORY_SIZE : historyIndex;

  for (uint8_t i = 0; i < total; i++) {
    if (history[i] == 0) continue;  // Skip invalid readings
    long diff = (long)history[i] - (long)currentDist;
    if (diff < 0) diff = -diff;     // Absolute value
    if (diff <= SIMILARITY_CM) {
      count++;
    }
  }
  return count;
}

void loop() {
  delay(150);  // ~150ms between pings, 10 pings ≈ 1.5s window

  unsigned long dist = sonar[0].ping_cm();

  // Store in history buffer
  history[historyIndex] = dist;
  historyIndex = (historyIndex + 1) % HISTORY_SIZE;
  if (historyIndex == 0) historyFull = true;

  Serial.print("dist=");
  Serial.print(dist);
  Serial.print("cm ");

  // Only check consensus if current reading is valid AND below threshold
  if (dist != 0 && dist < THRESHOLD_CM) {
    uint8_t agreement = countSimilar(dist);
    Serial.print("agree=");
    Serial.print(agreement);
    Serial.print("/");
    Serial.print(HISTORY_SIZE);

    if (agreement >= MIN_AGREEMENT) {
      Serial.print(" [TRIGGERED]");
      digitalWrite(TRIGGER_PIN, HIGH);
      delay(PULSE_MS);
      digitalWrite(TRIGGER_PIN, LOW);
    }
  }
  Serial.println();
}
