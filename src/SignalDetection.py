import numpy as np
import scipy

# groupwork

class SignalDetection:
   def __init__(self, hits, misses, falseAlarms, correctRejections):
       self.hits = hits
       self.misses = misses
       self.falseAlarms = falseAlarms
       self.correctRejections = correctRejections

   def hit_rate(self):
        if self.hits + self.misses == 0:
            return 0.5  # prevent undefined case
        return (self.hits) / (self.hits + self.misses)
  
   def false_alarm_rate(self):
        if self.falseAlarms + self.correctRejections == 0:
            return 0.5  # prevent undefined case
        return (self.falseAlarms) / (self.falseAlarms + self.correctRejections)
  
   def d_prime(self):
       hit_rate = self.hit_rate()
       false_alarm_rate = self.false_alarm_rate()
       return scipy.stats.norm.ppf(hit_rate) - scipy.stats.norm.ppf(false_alarm_rate)
  
   def criterion(self):
       hit_rate = self.hit_rate()
       false_alarm_rate = self.false_alarm_rate()
       return -0.5 * (scipy.stats.norm.ppf(hit_rate) + scipy.stats.norm.ppf(false_alarm_rate))
  
   def compute(self):
       return self.d_prime(), self.criterion()