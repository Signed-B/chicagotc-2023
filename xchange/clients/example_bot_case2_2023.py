#!/usr/bin/env python

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import asyncio
import json

PARAM_FILE = "params.json"


class OptionBot(UTCBot):
    """
    An example bot that reads from a file to set internal parameters during the round
    """

    async def handle_round_started(self):
        await asyncio.sleep(0.1)
        asyncio.create_task(self.handle_read_params())

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")
        # Competition event messages
        if kind == "generic_msg":
            msg = update.generic_msg.message
            print(msg)

    async def handle_read_params(self):
        while True:
            try:
                self.params = json.load(open(PARAM_FILE, "r"))
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)
    ########Boof stuff does not work###################
    #class OptionBot(UTCBot):
#     """
#     An example bot that reads from a file to set internal parameters during the round
#     """

#     async def handle_round_started(self):
#         await asyncio.sleep(0.1)
#         asyncio.create_task(self.handle_read_params())

#     async def handle_exchange_update(self, update: pb.FeedMessage):
#         kind, _ = betterproto.which_one_of(update, "msg")
#         # Competition event messages
#         if kind == "generic_msg":
#             msg = update.generic_msg.message
#             print(msg)
#         else:
#             market_data = update.market_data
#             stock_price = market_data.stock_price
#             option_positions = market_data.option_positions
#             hedge_quantity = self.delta_hedging(stock_price, option_positions)
#             if hedge_quantity > 0:
#                 await self.submit_order(pb.OrderSpec(
#                     action=pb.OrderAction.BUY,
#                     order_type=pb.OrderType.MARKET,
#                     stock_quantity=int(hedge_quantity),
#                 ))
#             elif hedge_quantity < 0:
#                 await self.submit_order(pb.OrderSpec(
#                     action=pb.OrderAction.SELL,
#                     order_type=pb.OrderType.MARKET,
#                     stock_quantity=int(-hedge_quantity),
#                 ))

#     async def handle_read_params(self):
#         while True:
#             try:
#                 self.params = json.load(open(PARAM_FILE, "r"))
#             except:
#                 print("Unable to read file " + PARAM_FILE)

#             await asyncio.sleep(1)
            
#     def delta_hedging(self, stock_price, option_positions):
#         delta = 0
#         for option in option_positions:
#             if option.option_type == pb.OptionType.CALL:
#                 option.delta = delta('c', stock_price, option.strike, option.days_to_expiry, 0.02, 0.2)

#         hedge_quantity = -delta
#         return hedge_quantity


if __name__ == "__main__":
    start_bot(OptionBot)
