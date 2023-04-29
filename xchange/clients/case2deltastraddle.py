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

DAYS_IN_ROUND = 30
DAYS_IN_GAME = 600
TICK_SIZE = 0.1
CONTRACTS = ['SPY'] + [f'SPY{65 + i*5}C' for i in range(15)] + [f'SPY{65 + 5*i}P' for i in range(15)]


class OptionBot(UTCBot):
    """
    An example bot that reads from a file to set internal parameters during the round
    """

    async def handle_round_started(self):
        await asyncio.sleep(0.1)

        self.vol = .28 # FIX THIS

        self._day = 0

        self._best_bid: Dict[str, float] = defaultdict(lambda: 0)

        self._best_ask: Dict[str, float] = defaultdict(lambda: 0)

        self.__orders: DefaultDict[str, Dict[str, Tuple[pb.OrderSpec, float]]] = defaultdict(lambda: ("", 0))

        self.__ordertimes: Dict[str, float] = defaultdict(lambda: np.inf)

        self.portfolio_delta = 0
        self.portfolio_gamma = 0
        self.portfolio_theta = 0
        self.portfolio_vega = 0

        self.asset_volatilities = {}
        self.asset_ivs = {}
        self.asset_deltas = {}
        self.asset_gammas = {}
        self.asset_vegas = {}
        self.asset_thetas = {}
        self.underlying_price = 50 # trash value to start

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
            try:
                ub_oid, ub_price = self.__orders["underlying_bid_{}".format(asset)]
                ua_oid, ua_price = self.__orders["underlying_ask_{}".format(asset)]


                if self._best_bid[asset] == 0 or self._best_ask[asset] == 0:                
                    await asyncio.sleep(1)
                    # print(asset, "waiting for best bid/ask")
                    continue

                # bidq = 10
                # askq = 10
                # if asset in self.positions and self.positions[asset] > 0:
                #     bidq = min(100, max(100 - self.positions[asset], 0))
                # if asset in self.positions and self.positions[asset] < 0:
                #     askq = min(100, max(100 + self.positions[asset], 0))
                
                time_to_expiry = (21.0 + self._day) / 252.0
                vol = self.compute_vol_estimate()

                requests = []

                strike = 100 # AT THE MONEY
                        
                strike = 5 * round(strike / 5)

                call_name = f"SPY{strike}C"
                put_name = f"SPY{strike}P"

                # call_theo = self.compute_option_price(
                #     # flag, self.underlying_price, strike, time_to_expiry, self.asset_ivs[asset_name]
                #     "C", self.underlying_price, strike, time_to_expiry, vol
                # )
                # put_theo = self.compute_option_price(
                #     # flag, self.underlying_price, strike, time_to_expiry, self.asset_ivs[asset_name]
                #     "P", self.underlying_price, strike, time_to_expiry, vol
                # )
                requests.append(
                    self.place_order(
                        call_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.BID,
                        15,
                        self._best_bid[call_name] - TICK_SIZE,
                    )
                )

                responses = await asyncio.gather(*requests)
                for r in responses:
                    self.__orders[f"underlying_bid_{asset}"] = (r.order_id, self._best_bid[call_name] - TICK_SIZE)
                    self.__ordertimes[r.order_id] = time.time()

                requests = []
                
                requests.append(
                    self.place_order(
                        put_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.BID,
                        15,
                        self._best_bid[put_name] - TICK_SIZE,
                    )
                )

                responses = await asyncio.gather(*requests)
                for r in responses:
                    self.__orders[f"underlying_ask_{asset}"] = (r.order_id, self._best_bid[put_name] - TICK_SIZE)
                    self.__ordertimes[r.order_id] = time.time()
                

                await self.hedge()
                
                self.update_portfolio_info()

                await self.kill_old()
            except Exception as e:
                print("error in trade")
                print(e)

    async def hedge(self):
        # HEDGING CODE
        delta = self.portfolio_delta
        requests = []
        if(self.portfolio_delta < 0):
            while(delta < 0):
                requests.append(
                    self.place_order("SPY", pb.OrderSpecType.LIMIT, pb.OrderSpecSide.BID, -int(max(delta, -1000)), self._best_ask["SPY"] + TICK_SIZE)
                )
                delta += 1000
        elif(self.portfolio_delta > 0):
            while(delta > 0):
                requests.append(
                    await self.place_order("SPY", pb.OrderSpecType.LIMIT, pb.OrderSpecSide.ASK, int(min(1000, delta)), self._best_bid["SPY"] - TICK_SIZE)
                )
                delta -= 1000

        resp = await asyncio.gather(*requests)
        for r in resp:
            self.__orders[f"underlying_ask_SPY"] = (r.order_id, self._best_ask["SPY"] + TICK_SIZE if self.portfolio_delta > 0 else self._best_bid["SPY"] - TICK_SIZE)
            self.__ordertimes[r.order_id] = time.time()




    def compute_vol_estimate(self) -> float:
        """
        This function is used to provide an estimate of underlying's volatility. Because this is
        an example bot, we just use a placeholder value here. We recommend that you look into
        different ways of finding what the true volatility of the underlying is.
        """
        
        # if (self.current_day == 0):
        #     return

        return self.vol

    # GIVEN
    def d1(self, S,K,T,r,sigma):
        return(np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))

    def d2(self, S,K,T,r,sigma):
        return self.d1(S,K,T,r,sigma)-sigma*np.sqrt(T)

    def bs_call(self, S,K,T,r,sigma):
        return S*norm.cdf(self.d1(S,K,T,r,sigma))-K*np.exp(-r*T)*norm.cdf(self.d2(S,K,T,r,sigma))

    def bs_put(self, S,K,T,r,sigma):
        return K*np.exp(-r*T)-S+self.bs_call(S,K,T,r,sigma)

    def iv_call(self, S,K,T,r,C):
        return implied_volatility(C, S, K, T, r, 'c')
        # return .5
        # return max(0, fsolve((lambda sigma: np.abs(self.bs_call(S,K,T,r,sigma) - C)), [1])[0])
                        
    def iv_put(self, S,K,T,r,P):
        return implied_volatility(P, S, K, T, r, 'p')
        # return .5
        # return max(0, fsolve((lambda sigma: np.abs(self.bs_put(S,K,T,r,sigma) - P)), [1])[0])

    def delta_call(self, S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * norm.cdf(self.d1(S,K,T,0,sigma))

    def gamma_call(self, S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma))/(S * sigma * np.sqrt(T))

    def vega_call(self, S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma)) * S * np.sqrt(T)

    def theta_call(self, S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * S * norm.pdf(self.d1(S,K,T,0,sigma)) * sigma/(2 * np.sqrt(T))

    def delta_put(self, S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * (norm.cdf(self.d1(S,K,T,0,sigma)) - 1)

    def gamma_put(self, S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma))/(S * sigma * np.sqrt(T))

    def vega_put(self, S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma)) * S * np.sqrt(T)

    def theta_put(self, S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * S * norm.pdf(self.d1(S,K,T,0,sigma)) * sigma/(2 * np.sqrt(T))

    def compute_option_price(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:

        # By default assign to the previous price in case we encounter an error
        price = self.asset_quotes[f"SPY{strike_px}{flag}"]
        if(flag == "C"):
            try:
                price = round(self.bs_call(underlying_px, strike_px, time_to_expiry, 0, volatility), 1)
            except:
                price = price
            return price

        else:
            try:
                price = round(self.bs_put(underlying_px,
                              strike_px, time_to_expiry, 0, volatility), 1)
            except:
                price = price
            return price

    def compute_option_iv(
        self,
        price: float,
        S: float,
        K: float,
        t: float,
        flag: str,
        asset_name: str,
    ) -> float:
        """
        This function is used to provide an estimate of underlying's volatility.

        price (float) – the Black-Scholes option price
        S (float) – underlying asset price
        K (float) – strike price
        t (float) – time to expiration in years
        flag (str) – ‘c’ or ‘p’ for call or put.
        """
        iv = self.asset_volatilities[asset_name]
        if(flag == "C"):
            try:
                iv = implied_volatility(float(price), S, K, t, 0.0, 'c')
                self.asset_ivs[asset_name].append(iv)
                if(len(self.asset_ivs[asset_name]) > 30):
                    self.asset_ivs[asset_name].pop(0)
            except:
                iv = iv
            return iv

        else:
            try:
                iv = implied_volatility(float(price), S, K, t, 0.0, 'p')
                self.asset_ivs[asset_name].append(iv)
                if(len(self.asset_ivs[asset_name]) > 30):
                    self.asset_ivs[asset_name].pop(0)
            except:
                iv = iv
            return iv

    def compute_option_delta(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        price: float,
    ) -> float:
        """
        - flag should be 'c' or 'p' (string)
        - prices are floats
        - time to expiry must be in years
        """

        if(flag == "C"):
            return round(self.delta_call(underlying_px, strike_px, time_to_expiry, price), 3)

        else:
            return round(self.delta_put(underlying_px, strike_px, time_to_expiry, price), 3)

    def compute_option_gamma(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        price: float,
    ) -> float:
        """
        - flag should be 'c' or 'p' (string)
        - prices are floats
        - time to expiry must be in years
        """

        if(flag == "C"):
            return round(self.gamma_call(underlying_px, strike_px, time_to_expiry, price), 5)

        else:
            return round(self.gamma_put(underlying_px, strike_px, time_to_expiry, price), 5)

    def compute_option_theta(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        price: float,
    ) -> float:
        """
        - flag should be 'c' or 'p' (string)
        - prices are floats
        - time to expiry must be in years
        """

        if(flag == "C"):
            return round(self.theta_call(underlying_px, strike_px, time_to_expiry, price), 5)

        else:
            return round(self.theta_put(underlying_px, strike_px, time_to_expiry, price), 5)

    def compute_option_vega(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        price: float,
    ) -> float:
        """
        - flag should be 'c' or 'p' (string)
        - prices are floats
        - time to expiry must be in years
        """

        if(flag == "C"):
            return round(self.vega_call(underlying_px, strike_px, time_to_expiry, price), 5)

        else:
            return round(self.vega_put(underlying_px, strike_px, time_to_expiry, price), 5)
    
    async def update_portfolio_info(self):
        """
        This function creates new option prices and greeks based on most recent market data.
        """

        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        # Update our fair values and greeks for each asset, with new asset price/time to expiry/volatility
        for strike in [65 + 5 * i for i in range(0, 15)]:
            for flag in ["C", "P"]:
                asset_name = f"SPY{strike}{flag}"

                time_to_expiry = (21.0 + self._day / 252.0)

                price = self.compute_option_price(
                    flag, self.underlying_price, strike, time_to_expiry, self.asset_volatilities[
                        asset_name]
                )
                delta = self.compute_option_delta(
                    flag, self.underlying_price, strike, time_to_expiry, price
                )
                gamma = self.compute_option_gamma(
                    flag, self.underlying_price, strike, time_to_expiry, price
                )
                theta = self.compute_option_theta(
                    flag, self.underlying_price, strike, time_to_expiry, price
                )
                vega = self.compute_option_vega(
                    flag, self.underlying_price, strike, time_to_expiry, price
                )
                iv = self.compute_option_iv(
                    price, self.underlying_price, strike, time_to_expiry, flag, asset_name
                )

                # self.asset_quotes[asset_name] = price
                self.asset_deltas[asset_name] = delta
                self.asset_gammas[asset_name] = gamma
                self.asset_thetas[asset_name] = theta
                self.asset_vegas[asset_name] = vega
                self.asset_ivs[asset_name] = iv

                total_delta += delta * float(self.positions[asset_name])
                total_gamma += gamma * float(self.positions[asset_name])
                total_theta += theta * float(self.positions[asset_name])
                total_vega += vega * float(self.positions[asset_name])


        self.portfolio_delta = int((total_delta) + self.positions["SPY"])
        self.portfolio_gamma = round(gamma, 3)
        self.portfolio_theta = round(theta, 3)
        self.portfolio_vega = round(vega, 3)

    def print_portfolio_greeks(self):
        print("=== Greeks ===")
        print("         CU        Limit")
        print("Delta: ", self.portfolio_delta, "      +/- 2000")
        print("Gamma: ", self.portfolio_gamma, "     +/- 5000")
        print("Vega: ", self.portfolio_vega, "      +/- 1000000")
        print("Theta:  ", abs(self.portfolio_theta), "        500000")
        return

    async def printer(self):
        while True:
            await asyncio.sleep(1)

            print("POS", self.positions)
            try:
                self.print_portfolio_greeks()
            except Exception as e:
                print("error in printer")
                print(e)
    

def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))             



 


if __name__ == "__main__":
    start_bot(OptionBot)
