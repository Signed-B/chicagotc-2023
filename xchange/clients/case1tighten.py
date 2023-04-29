#!/usr/bin/env python

from collections import defaultdict
from typing import DefaultDict, Dict, Tuple
from utc_bot import UTCBot, start_bot
import math
import proto.utc_bot as pb
import betterproto
import asyncio
import re
import time
import numpy as np
import copy as cp

DAYS_IN_MONTH = 21
DAYS_IN_YEAR = 252
INTEREST_RATE = 0.02
NUM_FUTURES = 14
TICK_SIZE = 0.00001
FUTURE_CODES = [chr(ord('A') + i) for i in range(NUM_FUTURES)] # Suffix of monthly future code
CONTRACTS = ['SBL'] +  ['LBS' + c for c in FUTURE_CODES] + ['LLL']

class Case1Bot(UTCBot):
    """
    An example bot
    """
    etf_suffix = ''
    async def create_etf(self, qty: int):
        '''
        Creates qty amount the ETF basket
        DO NOT CHANGE
        '''
        if len(self.etf_suffix) == 0:
            return pb.SwapResponse(False, "Unsure of swap")
        return await self.swap("create_etf_" + self.etf_suffix, qty)

    async def redeem_etf(self, qty: int):
        '''
        Redeems qty amount the ETF basket
        DO NOT CHANGE
        '''
        if len(self.etf_suffix) == 0:
            return pb.SwapResponse(False, "Unsure of swap")
        return await self.swap("redeem_etf_" + self.etf_suffix, qty) 
    
    async def days_to_expiry(self, asset):
        '''
        Calculates days to expiry for the future
        '''
        future = ord(asset[-1]) - ord('A')
        expiry = 21 * (future + 1)
        return self._day - expiry

    async def handle_exchange_update(self, update: pb.FeedMessage):
        '''
        Handles exchange updates
        '''
        kind, _ = betterproto.which_one_of(update, "msg")
        #Competition event messages
        if kind == "generic_msg":
            msg = update.generic_msg.message
            
            # Used for API DO NOT TOUCH
            if 'trade_etf' in msg:
                self.etf_suffix = msg.split(' ')[1]
                
            # Updates current weather
            if "Weather" in update.generic_msg.message:
                msg = update.generic_msg.message
                weather = float(re.findall("\d+\.\d+", msg)[0])
                self._weather_log.append(weather)
                
            # Updates date
            if "Day" in update.generic_msg.message:
                self._day = int(re.findall("\d+", msg)[0])
                print('day', self._day)
                            
            # Updates positions if unknown message (probably etf swap)
            # else:
            #     resp = await self.get_positions()
            #     if resp.ok:
            #         self.positions = resp.positions
            #         # print("POS", resp.positions)

            # NOTE: NO ETF ARBITRAGE, ETF SWAPS, OR ETF CREATION/REDEMPTION.
            #       All opportunities will be closed by competitors already, not worth pursuing.
            #       Dedicate all bandwidth to market making.
                    
        elif kind == "market_snapshot_msg":
            for asset in CONTRACTS:
                book = update.market_snapshot_msg.books[asset]
                self._best_bid[asset] = float(book.bids[0].px)
                self._best_ask[asset] = float(book.asks[0].px)
                # print(self._best_bid[asset], self._best_ask[asset])
        
        resp = await self.get_positions()
        if resp.ok:
            self.positions = resp.positions
            print("POS", resp.positions)
            


    async def handle_round_started(self):
        ### Current day
        self._day = 0
        ### Best Bid in the order book
        self._best_bid: Dict[str, float] = defaultdict(
            lambda: 0
        )
        ### Best Ask in the order book
        self._best_ask: Dict[str, float] = defaultdict(
            lambda: 0
        )
        ### Order book for market making
        self.__orders: DefaultDict[str, Tuple[str, float]] = defaultdict(
            lambda: ("", 0)
        )
        self.__ordertimes: DefaultDict[float] = defaultdict(
            lambda: np.inf
        )
        self._fair_price: DefaultDict[str, float] = defaultdict(
            lambda: ("", 0)
        )
        self._spread: DefaultDict[str, float] = defaultdict(
            lambda: ("", 0)
        )

        self._quantity: DefaultDict[str, int] = defaultdict(
            lambda: ("", 0)
        )
        
        ### List of weather reports
        self._weather_log = []
        
        await asyncio.sleep(.5)
        ###
        ### TODO START ASYNC FUNCTIONS HERE
        ###
        
        # Starts market making for each asset
        for asset in CONTRACTS:
            asyncio.create_task(self.make_market_asset(asset))


    # eliminate old orders if unfilled: don't get thrown against the wall if price fluctuates
    async def kill_old(self):
        for id in cp.copy(self.__ordertimes):
            if time.time() - self.__ordertimes[id] > 3:
                r = await self.cancel_order(id)
                if r.ok or r.message == 'order ID not present in market':
                    if id in self.__ordertimes: self.__ordertimes.pop(id)
                # print('CANNED', id, r.message)

    

    async def make_market_asset(self, asset: str):
        while self._day <= DAYS_IN_YEAR:
            try:
                ## Old prices
                ub_oid, ub_price = self.__orders["underlying_bid_{}".format(asset)]
                ua_oid, ua_price = self.__orders["underlying_ask_{}".format(asset)]


                if self._best_bid[asset] == 0 or self._best_ask[asset] == 0:                
                    await asyncio.sleep(1)
                    print(asset, "waiting for best bid/ask")
                    continue
                

                pennywidth = TICK_SIZE
                bidq = 10
                askq = 10
                if asset in self.positions and self.positions[asset] > 0:
                    bidq = min(100, max(100 - self.positions[asset], 0))
                if asset in self.positions and self.positions[asset] < 0:
                    askq = min(100, max(100 + self.positions[asset], 0))
                # calculate order sizes: maintain zero directional exposure, maximum order size of 100. 
                
                # if no orders, place new ones.
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

                print(bid_px, ask_px)   

                
                # If the underlying price moved first, adjust the ask first to avoid self-trades
                if (bid_px + ask_px) > (ua_price + ub_price):
                    order = ["ask", "bid"]
                else:
                    order = ["bid", "ask"]

                # for orders already on the books, modify them if the price has moved (appropriate size)
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

                # kill old orders at every opportunity.
                await self.kill_old()
            # Error handling, don't let code stop b/c something dumb happened:
            except Exception as e:
                print("error in make market asset")
                print(e)





def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))             



if __name__ == "__main__":
    start_bot(Case1Bot)
