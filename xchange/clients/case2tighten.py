#!/usr/bin/env python

from collections import defaultdict
from typing import DefaultDict, Dict, Tuple
from utc_bot import UTCBot, start_bot
import math
import proto.utc_bot as pb
import betterproto
import asyncio
import json
import time
import copy as cp

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import pandas as pd
from py_vollib.ref_python.black_scholes import black_scholes
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black.greeks.analytical import delta, gamma, theta, vega

PARAM_FILE = "params.json"

DAYS_IN_ROUND = 30
DAYS_IN_GAME = 600
TICK_SIZE = 0.00001
CONTRACTS = ['SPY'] + [f'SPY{65 + i*5}C' for i in range(15)] + [f'SPY{65 + 5*i}P' for i in range(15)]


class OptionBot(UTCBot):
    """
    An example bot that reads from a file to set internal parameters during the round
    """

    async def handle_round_started(self):
        await asyncio.sleep(0.1)

        self._day = 0

        self._best_bid: Dict[str, float] = defaultdict(lambda: 0)

        self._best_ask: Dict[str, float] = defaultdict(lambda: 0)

        self.__orders: DefaultDict[str, Dict[str, Tuple[pb.OrderSpec, float]]] = defaultdict(lambda: ("", 0))

        self.__ordertimes: Dict[str, float] = defaultdict(lambda: np.inf)

        for asset in CONTRACTS:
            asyncio.create_task(self.trade(asset))
        # asyncio.create_task(self.handle_read_params())
        asyncio.create_task(self.printer())

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")
        # Competition event messages
        if kind == "generic_msg":
            msg = update.generic_msg.message
            print(msg)
            # resp = await self.get_positions()
            # if resp.ok:
            #     self.positions = resp.positions
        elif kind == "market_snapshot_msg":
            for asset in CONTRACTS:
                book = update.market_snapshot_msg.books[asset]
                self._best_bid[asset] = 0 if len(book.bids) == 0 else float(book.bids[0].px)
                self._best_ask[asset] = 1000 if len(book.asks) == 0 else float(book.asks[0].px)

        resp = await self.get_positions()
        if resp.ok:
            self.positions = resp.positions
        

    async def kill_old(self):
        for id in cp.copy(self.__ordertimes):
            if time.time() - self.__ordertimes[id] > 3:
                r = await self.cancel_order(id)
                if r.ok or r.message == 'order ID not present in market':
                    if id in self.__ordertimes: self.__ordertimes.pop(id)


    async def trade(self, asset: str):
        while self._day <= DAYS_IN_GAME:
            ub_oid, ub_price = self.__orders["underlying_bid_{}".format(asset)]
            ua_oid, ua_price = self.__orders["underlying_ask_{}".format(asset)]


            if self._best_bid[asset] == 0 or self._best_ask[asset] == 0:                
                await asyncio.sleep(1)
                continue

            pennywidth = 0
            bidq = 1
            askq = 1
            if asset in self.positions and self.positions[asset] > 0:
                bidq = min(100, max(100 - self.positions[asset], 0))
            if asset in self.positions and self.positions[asset] < 0:
                askq = min(100, max(100 + self.positions[asset], 0))
            
            if ub_oid == "" or ua_oid == "":
                bpr = self._best_bid[asset] - pennywidth
                apr = self._best_ask[asset] + pennywidth
                if bidq > 0:
                    rb = await self.place_order(
                        asset,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.BID,
                        bidq,
                        bpr,
                    )
                    self.__orders[f"underlying_bid_{asset}"] = (rb.order_id, bpr)
                    self.__ordertimes[rb.order_id] = time.time()
                if askq > 0:
                    ra = await self.place_order(
                        asset,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.ASK,
                        askq,
                        apr,
                    )
                    self.__orders[f"underlying_ask_{asset}"] = (ra.order_id, apr)
                    self.__ordertimes[ra.order_id] = time.time()
                
                print("initial done")
                continue
            

            bid_px = ub_price + pennywidth
            ask_px = ua_price - pennywidth

            # print(bid_px, ask_px)   

            
            # If the underlying price moved first, adjust the ask first to avoid self-trades
            if (bid_px + ask_px) > (ua_price + ub_price):
                order = ["ask", "bid"]
            else:
                order = ["bid", "ask"]

            for d in order:
                if d == "bid":
                    order_id = ub_oid
                    order_side = pb.OrderSpecSide.BID
                    order_px = bid_px
                    if bidq > 0:
                        r = await self.modify_order(
                            order_id = order_id,
                            asset_code = asset,
                            order_type = pb.OrderSpecType.LIMIT,
                            order_side = order_side,
                            qty = bidq,
                            px = round_nearest(order_px, TICK_SIZE), 
                        )
                        self.__orders[f"underlying_{d}_{asset}"] = (r.order_id, order_px)
                        self.__ordertimes[r.order_id] = time.time()
                else:
                    order_id = ua_oid
                    order_side = pb.OrderSpecSide.ASK
                    order_px = ask_px
                    if askq > 0:
                        r = await self.modify_order(
                            order_id = order_id,
                            asset_code = asset,
                            order_type = pb.OrderSpecType.LIMIT,
                            order_side = order_side,
                            qty = askq,
                            px = round_nearest(order_px, TICK_SIZE), 
                        )
                        self.__orders[f"underlying_{d}_{asset}"] = (r.order_id, order_px)
                        self.__ordertimes[r.order_id] = time.time()

                await self.kill_old()

    async def handle_read_params(self):
        while True:
            try:
                self.params = json.load(open(PARAM_FILE, "r"))
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)

    async def printer(self):
        while True:
            print("POS", self.positions)
            await asyncio.sleep(1)



    

    

def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))             



 


if __name__ == "__main__":
    start_bot(OptionBot)
