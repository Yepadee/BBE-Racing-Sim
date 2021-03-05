class Order(object):
    def __init__(self, bettor_id, competetor_id, order_type, odds, stake):
        self.bettor_id = bettor_id
        self.competetor_id = competetor_id
        self.order_type = order_type
        self.odds = odds
        self.stake = stake


class BettingMarket(object):
    def __init__(self):
        pass

    def add_order(self):
        pass