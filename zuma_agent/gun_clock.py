"""Conservative input timing based on the gun's firing/reloading state machine."""
from dataclasses import dataclass
import time


@dataclass
class GunClock:
    interval: float = .24
    last_fire: float = -1e9
    last_swap: float = -1e9

    def ready_at(self):
        return max(self.last_fire+self.interval,self.last_swap+.04)

    def fired(self,now):
        if now+1e-6 < self.ready_at():
            raise ValueError('Fire before the conservative reload window has ended')
        self.last_fire = now

    def wait_until_ready(self,clock=time.monotonic,sleep=time.sleep):
        # Windows clocks and timer wakeups can straddle a coarse tick. Check the
        # deadline again before input, rather than trusting a single sleep.
        while True:
            remaining=self.ready_at()-clock()
            if remaining<=0:
                return
            sleep(remaining+.002)

    def swapped(self,now):
        self.last_swap = now
