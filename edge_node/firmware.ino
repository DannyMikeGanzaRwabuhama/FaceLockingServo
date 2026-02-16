/*
 * ESP8266 Firmware - Distributed Vision-Control System (Phase 1)
 * --------------------------------------------------------------
 * Role: Edge Actuator
 * 1. Connects to WiFi & MQTT Broker.
 * 2. Subscribes to vision/<team_id>/movement.
 * 3. Moves Servo based on "MOVE_LEFT" or "MOVE_RIGHT" commands.
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// ==========================================
// CONFIGURATION
// ==========================================
const char* ssid = "RCA";
const char* password = "@RcaNyabihu2023";

const char* mqtt_server = "157.173.101.159";
const int mqtt_port = 1883;

const char* topic_movement = "vision/y3c_ajhm/movement";

const char* client_id = "esp8266_servo_node";

const int SERVO_PIN = D1;

// ==========================================
// GLOBALS
// ==========================================
WiFiClient espClient;
PubSubClient client(espClient);
Servo myServo;

int currentAngle = 90; // Start at center (90 degrees)

// ==========================================
// SETUP
// ==========================================
void setup() {
  Serial.begin(9600);
  delay(10);

  Serial.println("\n===== ESP8266 Vision Controller =====");

  // Initialize Servo
  myServo.attach(SERVO_PIN);
  myServo.write(currentAngle);
  Serial.println("Servo Initialized at 90 degrees");

  // Connect to Network
  setup_wifi();

  // MQTT Setup
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

// ==========================================
// WIFI SETUP
// ==========================================
void setup_wifi() {
  delay(10);
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA); // Station mode
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n✅ WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// ==========================================
// MQTT CALLBACK (Listen for Messages)
// ==========================================
void callback(char* topic, byte* payload, unsigned int length) {
  // Convert payload to String for easy parsing
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.print("📩 Msg on [");
  Serial.print(topic);
  Serial.print("]: ");
  Serial.println(message);

  // --- PARSING LOGIC ---
  // The Python script sends JSON: {"status": "MOVE_LEFT", ...}

  if (message.indexOf("MOVE_LEFT") >= 0) {
    // Face is to the LEFT, so rotate Servo LEFT (increase angle)
    currentAngle = currentAngle + 3;
    if (currentAngle > 180) currentAngle = 180;

    myServo.write(currentAngle);
    Serial.println("⬅️ Moving Servo LEFT");
  }
  else if (message.indexOf("MOVE_RIGHT") >= 0) {
    // Face is to the RIGHT, rotate Servo RIGHT (decrease angle)
    currentAngle = currentAngle - 3;
    if (currentAngle < 0) currentAngle = 0;

    myServo.write(currentAngle);
    Serial.println("➡️ Moving Servo RIGHT");
  }
  else if (message.indexOf("CENTERED") >= 0) {
    Serial.println("✅ Target Centered - Holding Position");
  }
  else if (message.indexOf("NO_FACE") >= 0) {
    Serial.println("❌ No Face Detected - Holding Position");
  }
}

// ==========================================
// RECONNECT LOOP
// ==========================================
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");

    // Attempt to connect with UNIQUE ID
    if (client.connect(client_id)) {
      Serial.println("connected!");

      // Subscribe to the specific team topic
      client.subscribe(topic_movement);
      Serial.print("Subscribed to: ");
      Serial.println(topic_movement);

    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

// ==========================================
// MAIN LOOP
// ==========================================
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
