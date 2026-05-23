import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import json
import time
import csv

# Default values
MARKET_ID = "XEUR"
DATE = "20191202"
MARKET_SEGMENT_ID = "688"
SECURITY_ID = "4128839"

# User defined values
if len(sys.argv) == 5:
    MARKET_ID = sys.argv[1]
    DATE = sys.argv[2]
    MARKET_SEGMENT_ID = sys.argv[3]
    SECURITY_ID = sys.argv[4]

# Input and output file paths
INPUT_FILE_PATH = f"data/{MARKET_ID}_{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_detailed.json"
OUTPUT_FILE_PATH = f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_lobster.csv"

# FIX protocol template IDs
ORDER_ADD = 13100
ORDER_MODIFY = 13101
ORDER_MODIFY_SAME_PRIORITY = 13106
ORDER_DELETE = 13102
ORDER_MASS_DELETE = 13103
PARTIAL_ORDER_EXECUTION = 13105
FULL_ORDER_EXECUTION = 13104

# Load the data
print("Loading data...")
tic = time.time()
if len(sys.argv) == 2:
    with open(sys.argv[1], "r") as fp:
        data = json.load(fp)
else:
    with open(INPUT_FILE_PATH, "r") as fp:
        data = json.load(fp)
tac = time.time()
print(f"Data loaded in {tac - tic:.2f} seconds")

# Key - Timestamp | Value - (What to do, (Price, Quantity, Side, ...))
# Alternatively with MsgSeqNum possibly: # Key - MsgSeqNum | Value - (Timestamp, What to do, (Price, ...))
instructions = {}

print("Processing data...")
tic = time.time()
# Downloading is done in parts, thus iterate over the downloaded parts
for i, part in enumerate(data):
    print(f"Processing part {i + 1}/{len(data)}")

    # Iterate over the transactions in the part
    for transaction_array in part["Transactions"]:
        for transaction in transaction_array:
            # MsgSeqNum = transaction["MessageHeader"]["MsgSeqNum"]

            # Handle the ORDER ADD message as simple ADD instruction
            if transaction["MessageHeader"]["TemplateID"] == ORDER_ADD:
                TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                # TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]

                Price = float(transaction["OrderDetails"]["Price"]) / 1e8
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]

                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("ADD", (Price, DisplayQty, Side)))
                else:
                    instructions[TrdRegTSTimeIn] = [("ADD", (Price, DisplayQty, Side))]

                # instructions[MsgSeqNum] = [(TrdRegTSTimePriority, "ADD", (Price, DisplayQty, Side))]

            # Handle the ORDER MODIFY message as simple DELETE and ADD instruction
            elif transaction["MessageHeader"]["TemplateID"] == ORDER_MODIFY:
                TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                # TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]

                # TrdRegTSPrevTimePriority = transaction["TrdRegTSPrevTimePriority"]
                PrevPrice = float(transaction["PrevPrice"]) / 1e8
                PrevDisplayQty = float(transaction["PrevDisplayQty"]) / 1e8

                Price = float(transaction["OrderDetails"]["Price"]) / 1e8
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]

                # TrdRegTSPrevTimePriority -- yields worse results for some reason
                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("DELETE", (PrevPrice, PrevDisplayQty, Side)))
                else:
                    instructions[TrdRegTSTimeIn] = [("DELETE", (PrevPrice, PrevDisplayQty, Side))]

                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("ADD", (Price, DisplayQty, Side)))
                else:
                    instructions[TrdRegTSTimeIn] = [("ADD", (Price, DisplayQty, Side))]

                # instructions[MsgSeqNum] = [(TrdRegTSTimePriority, "DELETE", (PrevPrice, PrevDisplayQty, Side)),
                #                            (TrdRegTSTimePriority, "ADD", (Price, DisplayQty, Side))]

            # Handle the ORDER MODIFY SAME PRIORITY message as simple DELETE and ADD instruction
            elif transaction["MessageHeader"]["TemplateID"] == ORDER_MODIFY_SAME_PRIORITY:
                # TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                # TransactTime = transaction["TransactTime"]

                PrevDisplayQty = float(transaction["PrevDisplayQty"]) / 1e8

                Price = float(transaction["OrderDetails"]["Price"]) / 1e8
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]

                if TrdRegTSTimePriority in instructions:
                    instructions[TrdRegTSTimePriority].append(("DELETE", (Price, PrevDisplayQty, Side)))
                    instructions[TrdRegTSTimePriority].append(("ADD", (Price, DisplayQty, Side)))
                else:
                    instructions[TrdRegTSTimePriority] = [("DELETE", (Price, PrevDisplayQty, Side)), ("ADD", (Price, DisplayQty, Side))]

                # instructions[MsgSeqNum] = [(TransactTime, "DELETE", (Price, PrevDisplayQty, Side)),
                #                            (TransactTime, "ADD", (Price, DisplayQty, Side))]

            # Handle the ORDER DELETE message as simple DELETE instruction
            elif transaction["MessageHeader"]["TemplateID"] == ORDER_DELETE:
                TrdRegTSTimeIn = transaction["TrdRegTSTimeIn"]
                # TrdRegTSTimePriority = transaction["OrderDetails"]["TrdRegTSTimePriority"]
                # TransactTime = transaction["TransactTime"]

                Price = float(transaction["OrderDetails"]["Price"]) / 1e8
                DisplayQty = float(transaction["OrderDetails"]["DisplayQty"]) / 1e8
                Side = transaction["OrderDetails"]["Side"]

                if TrdRegTSTimeIn in instructions:
                    instructions[TrdRegTSTimeIn].append(("DELETE", (Price, DisplayQty, Side)))
                else:
                    instructions[TrdRegTSTimeIn] = [("DELETE", (Price, DisplayQty, Side))]

                # instructions[MsgSeqNum] = [(TransactTime, "DELETE", (Price, DisplayQty, Side))]

            # Handle the ORDER MASS DELETE message as its own instruction
            elif transaction["MessageHeader"]["TemplateID"] == ORDER_MASS_DELETE:
                TransactTime = transaction["TransactTime"]

                if TransactTime in instructions:
                    instructions[TransactTime].append(("ORDER_MASS_DELETE", ()))
                else:
                    instructions[TransactTime] = [("ORDER_MASS_DELETE", ())]

                # instructions[MsgSeqNum] = [(TransactTime, "ORDER_MASS_DELETE", ())]

            # Handle the PARTIAL ORDER EXECUTION message as simple DELETE and ADD instruction
            elif transaction["MessageHeader"]["TemplateID"] == PARTIAL_ORDER_EXECUTION:
                TrdRegTSTimePriority = transaction["TrdRegTSTimePriority"]

                Price = float(transaction["Price"]) / 1e8
                LastPx = float(transaction["LastPx"]) / 1e8
                LastQty = float(transaction["LastQty"]) / 1e8
                Side = transaction["Side"]

                if TrdRegTSTimePriority in instructions:
                    instructions[TrdRegTSTimePriority].append(("PARTIAL_ORDER_EXECUTION", (LastPx, LastQty, Side)))
                else:
                    instructions[TrdRegTSTimePriority] = [("PARTIAL_ORDER_EXECUTION", (LastPx, LastQty, Side))]

                # instructions[MsgSeqNum] = [(TrdRegTSTimePriority, "PARTIAL_ORDER_EXECUTION", (LastPx, LastQty, Side))]

            # Handle the FULL ORDER EXECUTION message as simple DELETE instruction
            elif transaction["MessageHeader"]["TemplateID"] == FULL_ORDER_EXECUTION:
                TrdRegTSTimePriority = transaction["TrdRegTSTimePriority"]

                Price = float(transaction["Price"]) / 1e8
                LastPx = float(transaction["LastPx"]) / 1e8
                LastQty = float(transaction["LastQty"]) / 1e8
                Side = transaction["Side"]

                if TrdRegTSTimePriority in instructions:
                    instructions[TrdRegTSTimePriority].append(("FULL_ORDER_EXECUTION", (LastPx, LastQty, Side)))
                else:
                    instructions[TrdRegTSTimePriority] = [("FULL_ORDER_EXECUTION", (LastPx, LastQty, Side))]

                # instructions[MsgSeqNum] = [(TrdRegTSTimePriority, "FULL_ORDER_EXECUTION", (LastPx, LastQty, Side))]

tac = time.time()

max_index = len(instructions)
print(f"Processed {max_index} transactions in {tac - tic:.2f} seconds")
print("Processing done")

# Variable "data" is now useless, free up memory
del data

# # Sort the instructions by key
# instructions = dict(sorted(instructions.items()))

# Create true orderbook now
print("Creating orderbook...")

# Initialize data structures
lobster_buy = [[] for _ in range(max_index)]  # Buy side
lobster_sell = [[] for _ in range(max_index)]  # Sell side
timestamps = list(dict(sorted({k: v for k, v in instructions.items() if k is not None}.items())).keys())
timestamps += [timestamps[-1]] * (max_index - len(timestamps))
cancellations_buy = {}  # Cancellations on buy side
cancellations_sell = {}  # Cancellations on sell side
trades_buy = {}  # Trades on buy side
trades_sell = {}  # Trades on sell side
levels = 100

BUY_SIDE = 1  # FIX protocol defines buy side as 1
SELL_SIDE = 2  # FIX protocol defines sell side as 2

tic = time.time()
for i, (timestamp, array) in enumerate(instructions.items()):
    # Print progress
    if i % 50_000 == 0:
        print(f"Processing order {i}/{max_index}")

    # Copy previous state
    if i > 0:
        lobster_buy[i] = lobster_buy[i - 1].copy()
        lobster_sell[i] = lobster_sell[i - 1].copy()

    # Initialize cancellations and trades
    if timestamp not in cancellations_buy:
        cancellations_buy[timestamp] = 0
    if timestamp not in cancellations_sell:
        cancellations_sell[timestamp] = 0
    if timestamp not in trades_buy:
        trades_buy[timestamp] = 0
    if timestamp not in trades_sell:
        trades_sell[timestamp] = 0

    # Iterate over the instructions
    for j, value in enumerate(array):
        if value[0] == "DELETE" or value[0] == "FULL_ORDER_EXECUTION" or value[0] == "PARTIAL_ORDER_EXECUTION":
            price, display_qty, side = value[1]

            # Choose the correct side to be modified
            if side == BUY_SIDE:
                lobster = lobster_buy
            else:
                lobster = lobster_sell

            # Update cancellations and trades
            if value[0] == "DELETE":
                if side == BUY_SIDE:
                    cancellations_buy[timestamp] += 1
                else:
                    cancellations_sell[timestamp] += 1
            if value[0] == "FULL_ORDER_EXECUTION" or value[0] == "PARTIAL_ORDER_EXECUTION":
                if side == BUY_SIDE:
                    trades_buy[timestamp] += 1
                else:
                    trades_sell[timestamp] += 1

            # Find the order in the heap
            found = False
            for k, (p, q) in enumerate(lobster[i]):
                if p == price:
                    found = True
                    break

            # Delete the order
            if found:
                lobster[i][k] = (price, q - display_qty)
                if q - display_qty <= 0:
                    del lobster[i][k]

            # else:
            #     print(f"Order not found: {value}")
            #     print(f"Lobster: {lobster[i]}")

        if value[0] == "ADD":
            price, display_qty, side = value[1]

            # Choose the correct side to be modified
            if side == BUY_SIDE:
                lobster = lobster_buy
            else:
                lobster = lobster_sell

            for p, q in lobster[i]:
                if p == price:
                    q += display_qty
                    break
            else:  # For-Else
                if len(lobster[i]) < levels:
                    lobster[i].append((price, display_qty))
                elif side == BUY_SIDE:
                    # Find the worst price
                    worst_price = min(lobster_buy[i], key=lambda x: x[0])[0]
                    if price > worst_price:
                        # Replace the worst price
                        for k, (p, q) in enumerate(lobster_buy[i]):
                            if p == worst_price:
                                lobster_buy[i][k] = (price, display_qty)
                                break
                elif side == SELL_SIDE:
                    # Find the worst price
                    worst_price = max(lobster_sell[i], key=lambda x: x[0])[0]
                    if price < worst_price:
                        # Replace the worst price
                        for k, (p, q) in enumerate(lobster_sell[i]):
                            if p == worst_price:
                                lobster_sell[i][k] = (price, display_qty)
                                break

        # Order mass delete is defined as a special case of completely emptying the orderbook
        if value[0] == "ORDER_MASS_DELETE":
            # Delete all orders
            lobster_buy[i] = []
            lobster_sell[i] = []

tac = time.time()
print(f"Created orderbook in {tac - tic:.2f} seconds")

# Variable "instructions" is now useless, free up memory
del instructions

# Export to lobster csv
print("Preparing for export to CSV...")

# Sort the levels correspondingly to the price (depends on buy / sell, best buy price is highest, best sell lowest)
for i in range(max_index):
    lobster_buy[i] = sorted(lobster_buy[i], reverse=True)
    lobster_sell[i] = sorted(lobster_sell[i])

# Time,Ask Price 1, Ask Volume 1, Bid Price 1, Bid Volume 1, ...
levels = 30
lobster_header = "Time,"
for i in range(levels):
    lobster_header += f"Ask Price {i + 1},Ask Volume {i + 1},Bid Price {i + 1},Bid Volume {i + 1},"
lobster_header += "Cancellations Buy,Cancellations Sell,Trades Buy,Trades Sell"  # This must be saved for the future calculation of Cancellations Rate
lobster_header = lobster_header.split(",")

print("Exporting to CSV...")

tic = time.time()
# Export to CSV
with open(OUTPUT_FILE_PATH, "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(lobster_header)  # Write header

    for i in range(max_index):
        # row = [i]
        row = [timestamps[i]]  # Start with timestamp

        for level in range(levels):
            # Add sell levels
            if level < len(lobster_sell[i]):
                row.extend([f"{lobster_sell[i][level][0]:.8f}", f"{lobster_sell[i][level][1]:.8f}"])
            else:
                row.extend(["", ""])  # Empty values

            # Add buy levels
            if level < len(lobster_buy[i]):
                row.extend([f"{lobster_buy[i][level][0]:.8f}", f"{lobster_buy[i][level][1]:.8f}"])
            else:
                row.extend(["", ""])  # Empty values

        row.extend([cancellations_buy[timestamps[i]]])  # Add cancellations buy
        row.extend([cancellations_sell[timestamps[i]]])  # Add cancellations sell
        row.extend([trades_buy[timestamps[i]]])  # Add trades buy
        row.extend([trades_sell[timestamps[i]]])  # Add trades sell

        writer.writerow(row)  # Write row to CSV

toc = time.time()
print(f"Exported to CSV in {toc - tic:.2f} seconds")
