import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import time
from datetime import datetime
import json
import requests
import random

if len(sys.argv) != 5:
    print("Usage: python download_eobi.py <MARKET_ID> <DATE> <MARKET_SEGMENT_ID> <SECURITY_ID>")
    print("Example: python download_eobi.py XEUR 20210104 691 5315926")
    exit(1)

MARKET_ID = sys.argv[1]
DATE = sys.argv[2]
MARKET_SEGMENT_ID = sys.argv[3]
SECURITY_ID = sys.argv[4]

if not os.path.exists("a7token.txt"):
    print("Please create a file named 'a7token.txt' with your API token.")
    exit(1)

userId = "zapped99@ntis.zcu.cz"  # Change this to your user ID
with open("a7token.txt", "r") as file:  # Change this to your token file
    API_TOKEN = file.read().rstrip()

url = "https://a7.deutsche-boerse.com/api/v1/"
header = {"Authorization": "Bearer " + API_TOKEN}

# 1e9 is for converting seconds to nanoseconds
last_time = str(int(datetime.strptime(DATE, "%Y%m%d").replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1e9))

part = 0

if not os.path.exists("data"):
    os.makedirs("data")

fp = open(f"data/{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_detailed.json", "w")
fp.write("[\n")

while True:
    # Request for the first 1 million transact times
    request = f"eobi/{MARKET_ID}/{DATE}/{MARKET_SEGMENT_ID}/{SECURITY_ID}?mode=reference&limit=1000000&from={last_time}"
    request_detailed = f"eobi/{MARKET_ID}/{DATE}/{MARKET_SEGMENT_ID}/{SECURITY_ID}?mode=detailed&limit=1000000&from={last_time}"

    response = requests.get(url=url + request, headers=header)

    # Check if the response is empty
    if len(response.json()["TransactTimes"]) == 0:
        break

    # Request for the detailed transact times
    response_detailed = requests.get(url=url + request_detailed, headers=header)

    print(f"Status: {response.status_code} and {response_detailed.status_code}")
    print(f"Downloaded {len(response.json()['TransactTimes'])} transact times")

    # Update the "last time" variable for downloading in parts
    previous_last_time = last_time
    last_time = str(int(response.json()["TransactTimes"][-1]) + 1)

    # This should never happen, but just in case
    if previous_last_time > last_time:
        print("Time is not increasing!")

    # Check if the last time is the same as the previous last time -> downloaded all transact times
    if previous_last_time == last_time:
        break

    print(f"From {datetime.fromtimestamp(int(previous_last_time) / 1000000000)} to {datetime.fromtimestamp(int(last_time) / 1000000000)}")

    # Write the data to a file
    if response.status_code == 200 and response_detailed.status_code == 200:
        fp.write(json.dumps(response_detailed.json(), indent=1))
        fp.write(",")
    else:
        print("Connection error!")
        exit(1)

    part += 1
    # Politely wait for a random time between 5 and 15 seconds (to avoid bombarding the API with requests)
    time.sleep(5 + random.random() * 10)

print(f"Done downloading transact times in {part} parts")

fp.seek(fp.tell() - 1, 0)
fp.write("\n]")
fp.close()

print(f"Done writing to {MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_detailed.json")
