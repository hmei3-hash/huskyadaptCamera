/**
 * =============================================================
 *  BLE SERVER 1 — Temperature Sensor
 *  Board: ESP32
 *  Library: NimBLE-Arduino
 * 
 *  This device advertises as "TempSensor_1" and notifies the
 *  connected client with a temperature reading every 2 seconds.
 * =============================================================
 */

#include "NimBLEDevice.h"

// -----------------------------------------------------------
// UUIDs — must match exactly on the client
// Each server has a UNIQUE Service UUID so the client can
// tell them apart. The Characteristic UUID can be the same.
// -----------------------------------------------------------
#define SERVICE_UUID        "A1B2C3D4-E5F6-7890-ABCD-111111111111"
#define CHAR_TEMP_UUID      "A1B2C3D4-E5F6-7890-ABCD-AABBCCDD0001"

// -----------------------------------------------------------
// Globals
// -----------------------------------------------------------
NimBLEServer*         pServer         = nullptr;
NimBLECharacteristic* pTempChar       = nullptr;
bool                  clientConnected = false;

// Simulate a temperature reading (replace with real sensor code)
float readTemperature() {
    // TODO: Replace with actual sensor read, e.g. DHT22, DS18B20, etc.
    static float base = 22.0f;
    base += ((float)(random(-10, 11))) / 10.0f;  // small random drift
    base = constrain(base, 15.0f, 35.0f);
    return base;
}

// -----------------------------------------------------------
// Server Callbacks — track connection state
// -----------------------------------------------------------
class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) override {
        Serial.println("[Server1] Client connected!");
        clientConnected = true;
    }

    void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) override {
        Serial.println("[Server1] Client disconnected — restarting advertising...");
        clientConnected = false;
        NimBLEDevice::startAdvertising();
    }
};

// -----------------------------------------------------------
// Setup
// -----------------------------------------------------------
void setup() {
    Serial.begin(115200);
    Serial.println("[Server1] Starting BLE Server 1...");

    NimBLEDevice::init("TempSensor_1");

    // Create server and attach callbacks
    pServer = NimBLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    // Create service
    NimBLEService* pService = pServer->createService(SERVICE_UUID);

    // Create temperature characteristic with NOTIFY property
    pTempChar = pService->createCharacteristic(
        CHAR_TEMP_UUID,
        NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY
    );

    // Set an initial value
    pTempChar->setValue("0.0");

    // Start the service
    pService->start();

    // Start advertising
    NimBLEAdvertising* pAdvertising = NimBLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setName("TempSensor_1");
    pAdvertising->start();

    Serial.println("[Server1] Advertising started. Waiting for client...");
}

// -----------------------------------------------------------
// Loop — send temperature notification every 2 seconds
// -----------------------------------------------------------
void loop() {
    if (clientConnected) {
        float temp = readTemperature();

        // Format as string: e.g. "23.40"
        char buf[16];
        snprintf(buf, sizeof(buf), "%.2f", temp);

        pTempChar->setValue(buf);
        pTempChar->notify();  // push to client

        Serial.printf("[Server1] Notified temp: %s °C\n", buf);
    }

    delay(2000);
}
