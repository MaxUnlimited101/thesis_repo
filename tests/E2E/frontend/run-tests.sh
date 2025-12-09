#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}Emotion Dashboard E2E Test Setup${NC}"
echo -e "${BLUE}=================================${NC}\n"

# Navigate to test directory
cd "$(dirname "$0")"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo -e "\n${BLUE}Installing dependencies...${NC}"
    npm install
    
    echo -e "\n${BLUE}Installing Playwright browsers...${NC}"
    npx playwright install
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

# Check if educator server is running
echo -e "\n${BLUE}Checking if educator server is running...${NC}"
if curl -s http://localhost:8001/health > /dev/null; then
    echo -e "${GREEN}✓ Educator server is running${NC}"
else
    echo -e "${RED}⚠ Educator server is not running${NC}"
    echo -e "${RED}Please run the educator server before running tests.${NC}"
    exit 1
fi

# Run tests
echo -e "\n${BLUE}=================================${NC}"
echo -e "${BLUE}Running E2E Tests${NC}"
echo -e "${BLUE}=================================${NC}\n"

npm test

# Show report
echo -e "\n${BLUE}=================================${NC}"
echo -e "${BLUE}Test Results${NC}"
echo -e "${BLUE}=================================${NC}\n"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    echo -e "\n${BLUE}To view detailed report, run:${NC}"
    echo -e "  npm run test:report"
else
    echo -e "${RED}Some tests failed${NC}"
    echo -e "\n${BLUE}To view detailed report, run:${NC}"
    echo -e "  npm run test:report"
    echo -e "\n${BLUE}To debug tests, run:${NC}"
    echo -e "  npm run test:debug"
fi
