#!/usr/bin/env python

from ds_pipe.data_feed import main_feed
from ds_pipe.data_preprocessing import hello

if __name__ == "__main__":
    main_feed("challenge_edMachina.csv")
    hello()
