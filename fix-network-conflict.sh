#!/bin/bash

echo "🔧 Docker Network Conflict Resolution"
echo "===================================="
echo

echo "The error indicates a network subnet conflict. Here are your options:"
echo

echo "📋 **Option 1: Try the updated configuration (RECOMMENDED)**"
echo "-----------------------------------------------------------"
echo "I've already updated docker-compose.yml to use subnet 172.25.0.0/16"
echo "Try running:"
echo "  docker-compose up --build"
echo

echo "📋 **Option 2: Clean up existing networks**"
echo "------------------------------------------"
echo "If you still get conflicts, check for existing networks:"
echo "  docker network ls"
echo
echo "Remove conflicting networks (be careful!):"
echo "  docker network prune  # Remove unused networks"
echo "  docker network rm <network-name>  # Remove specific network"
echo

echo "📋 **Option 3: Use Docker's default networking**"
echo "----------------------------------------------"
echo "If you don't need custom networking, you can comment out the networks"
echo "section in docker-compose.yml and let Docker handle it automatically."
echo

echo "📋 **Option 4: Use a different subnet**"
echo "-------------------------------------"
echo "Edit docker-compose.yml and try one of these subnets:"
echo "  - subnet: 172.30.0.0/16"
echo "  - subnet: 172.31.0.0/16"
echo "  - subnet: 192.168.100.0/24"
echo "  - subnet: 10.10.0.0/16"
echo

echo "📋 **Option 5: Force recreation**"
echo "-------------------------------"
echo "Force Docker Compose to recreate everything:"
echo "  docker-compose down --volumes --remove-orphans"
echo "  docker-compose up --build --force-recreate"
echo

echo "🚨 **If you need to reset everything:**"
echo "------------------------------------"
echo "⚠️  WARNING: This will remove ALL Docker networks, containers, and volumes!"
echo "  docker system prune -a --volumes"
echo

echo "✅ **Current configuration:**"
echo "  Network: media-to-text-network"
echo "  Subnet: 172.25.0.0/16"
echo "  This subnet is less commonly used and should avoid conflicts."
echo

echo "🚀 **Try running this now:**"
echo "  docker-compose up --build"