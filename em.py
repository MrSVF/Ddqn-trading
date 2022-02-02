import numpy as np

class equity:
    def __init__(self):
        self.qty    = np.int32(0);
        self.price  = np.float64(0);
        self.e_val  = 0;
        self.profit = 0;
        pass
    
    def __del__(self):
        pass
    
    def add(self, trd_price : np.float64, trd_qty : np.int32):
        self.profit = np.float64(0)
        
        if(trd_qty > 0): #buy
            
            #close short
            if(self.qty < 0):
                op_qty = trd_qty
                if(op_qty > -self.qty): op_qty = -self.qty
                
                self.qty += op_qty
                
                if(trd_price >= np.float64(0)): self.profit = (self.price - trd_price) * op_qty;
                
                self.e_val += self.profit
                trd_qty -= op_qty

            #open/add long
            if(trd_qty > 0):
                if(trd_price >= np.float64(0)): self.price = (self.qty * self.price + trd_qty * trd_price) / (self.qty + trd_qty)
                self.qty += trd_qty;
            
        elif(trd_qty < 0): #sell
            
            #close long
            if(self.qty > 0):
                op_qty = trd_qty;
                if(op_qty < -self.qty): op_qty = -self.qty;
                
                self.qty += op_qty;
                
                if(trd_price >= np.float64(0)): self.profit = (self.price - trd_price) * op_qty;
                
                self.e_val += self.profit
                trd_qty -= op_qty
                
            #open/add short
            if(trd_qty < 0):
                if(trd_price >= np.float64(0)): self.price = (self.qty * self.price + trd_qty * trd_price) / (self.qty + trd_qty)
                self.qty += trd_qty;
        pass
#def class equity
