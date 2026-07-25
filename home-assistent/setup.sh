#!/bin/bash
set -e

echo "🏠 Criando estrutura de diretórios..."

mkdir -p config/homeassistant
mkdir -p config/mosquitto
mkdir -p config/zigbee2mqtt
mkdir -p data/mariadb
mkdir -p data/mosquitto
mkdir -p data/nodered
mkdir -p data/influxdb
mkdir -p data/grafana
mkdir -p logs/mosquitto

echo "📝 Criando configuração do Mosquitto..."
cat > config/mosquitto/mosquitto.conf << 'EOF'
listener 1883
listener 9001
protocol websockets
allow_anonymous true
persistence true
persistence_location /mosquitto/data/
log_dest file /mosquitto/log/mosquitto.log
log_dest stdout
EOF

echo "📝 Criando configuração do Zigbee2MQTT..."
cat > config/zigbee2mqtt/configuration.yaml << 'EOF'
homeassistant: true
permit_join: true
mqtt:
  base_topic: zigbee2mqtt
  server: mqtt://mosquitto:1883
serial:
  port: /dev/ttyUSB0   # ajuste para seu adaptador
frontend:
  port: 8080
advanced:
  log_output:
    - console
EOF

echo ""
echo "✅ Setup concluído! Estrutura criada:"
find . -type d | sort | sed 's|[^/]*/|  |g'

echo ""
echo "🚀 Para subir os serviços:"
echo "   docker compose up -d"
echo ""
echo "🌐 Acessos após iniciar:"
echo "   Home Assistant  → http://localhost:8123"
echo "   Node-RED        → http://localhost:1880"
echo "   Grafana         → http://localhost:3000  (admin/adminpassword)"
echo "   InfluxDB        → http://localhost:8086  (admin/adminpassword)"
echo "   Zigbee2MQTT     → http://localhost:8080"
echo "   MQTT Broker     → localhost:1883"
