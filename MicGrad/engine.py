import math 
import numpy as np 

class Value:
    def __init__(self,data,_children=(),_op='',label = '')-> None:
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0
        self._backword = lambda : None 
    def __repr__(self) -> str:
        return f"Value(data={self.data})" 
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other),'+')
        def _backword():
            '''
            grad = local_gradiant * global_gradiant  
            global_gradiant -> dl/dout 
            local_gradiant  -> dout/dx -> (1)
            '''
            self.grad +=  1  *  out.grad 
            other.grad += 1  *  out.grad
        out._backword = _backword
        
        return out
    
    def __radd__(self,other):
        return self * other
    
   
       
    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,),'exp')
        def _backword():
            '''
            grad = local_gradiant * global_gradiant  
            global_gradiant -> dl/dout 
            local_gradiant  -> dy/dx exp(x) -> dy/dx exp(x)
            '''
            self.grad  += out.data * out.grad
        out._backword = _backword
        return out

    def __mul__(self,other):
         other = other if isinstance(other, Value) else Value(other)
         out = Value(self.data * other.data, (self,other),'*')
        
         def _backword():
            '''
            grad = local_gradiant * global_gradiant  
            global_gradiant -> dl/dout 
            local_gradiant  -> dout/dx -> other.data
            '''
            self.grad  +=  other.data * out.grad
            other.grad += self.data * out.grad
         out._backword = _backword
        
         return out
    def __pow__(self, other):
         assert isinstance(other, (int, float)), "only supporting int/float powers for now"

         out = Value(self.data**other, (self,), f'**{other}')

         def _backward():
            '''
            grad = local_gradiant * global_gradiant  
            global_gradiant -> dl/dout -> out.grad
            local_gradiant  -> dout/dx -> dy/dx(x^n) -> n(x^(n-1))
            '''
            self.grad += other * (self.data ** (other - 1)) * out.grad
            
         out._backward = _backward

         return out
    
    def __rmul__(self,other): # to Hendel 1 * Value case
        return self * other
    
    def __truediv__(self, other): # to 
        return self * other**-1

    def __neg__(self): 
        return self * -1


    def __sub__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self,other),'-')
        def _backword():
            '''
            grad = local_gradiant * global_gradiant  
            global_gradiant -> dl/dout 
            local_gradiant  -> dout/dx -> (-1)
            '''
            self.grad +=  -1  *  out.grad
            other.grad += -1  *  out.grad
        out._backword = _backword

        return out
    def __rsub__(self,other):
        return self * other
    
    def tanh(self):
        '''
        Just backword the backword Derivative
        '''
        x = self.data
        tanh = (np.exp(2*x) -1) / (np.exp(2*x) + 1)
        out = Value(tanh,(self,),'Tanh')    
        def _backword():
            self.grad  += (1 - (tanh ** 2)) * out.grad
        out._backword = _backword

        return out
    
    def sigmoid(self):
        x = self.data
        sigmoid = 1 /  1 +(math.exp(-x))
        out = Value(sigmoid,(self,),'Sigmoid')    
        def _backword():
            self.grad  += (sigmoid * (1 - sigmoid)) * out.grad
        out._backword = _backword
        return out

    def relu(self):
        if self.data < 0 :
            out = Value(0 , (self,), 'ReLU')
        else:
            out = Value(self.data , (self,), 'ReLU')
     
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def backword(self):
        Node_Topo_order = []
        seen = set()
        def Topo_Sort(Node):
            if Node not in seen:
                seen.add(Node)
                for v in Node._prev:
                    Topo_Sort(v)
                Node_Topo_order.append(Node)

        Topo_Sort(self)
        self.grad = 1
       
        for n in reversed(Node_Topo_order):
            n._backword()
