# Overview 

The goal of this project is to create an AI which can *profitably* trade on the various crypto markets in polymarket.com. We will start training it to perform on the BTC 5-minute market and potentially expand to other markets. 

## The Polymarket BTC 5-Minute Market

The URL for this is: https://polymarket.com/event/btc-updown-5m-1773712200 - though the ending will change every five minutes to match a Unix timestamp for that five-minute period.

This market works in the following manner:
    - Every five minutes, the price of Bitcoin is noted. This is the "price to beat" noted on the page.
    - At the end of the five minutes, the market resolves to either a Bitcoin price higher or lower than the price to beat. In case of ties, it resolves to higher. On the website, it is simply up and down.
    - At the conclusion of the five minutes, all participants who have purchased shares exchange money. If the market resolves up, all the people who sold "up" shares during the five minutes pay $1 per share to each of the buyers of those shares. If the market resolves down, all the people who sold "down" shares during the five minutes pay $1 per share to each of the buyers of those shares. Owners of "up" shares when the market results down, or owners of "down" shares when the market results up, get nothing, and the sellers of those shares get to keep the amount of money that was paid to them for those shares.
    - During the five-minute period, all market participants are free to purchase or sell up or down shares at the highest bid price (if they are selling shares), or buy at the lowest ask price (if they are buying).
    - Market participants may also add orders to the order book, for other market participants to potentially fill. Of course, if it's an ask order, it must be higher than the current lowest ask order in either the up or down shares. Conversely, it must be a lower bid amount for purchasing either up or down shares.
    - Those who add orders to the order book are called "market makers", and if their order is filled, they receive a bonus. Conversely, those who purchase orders at the market price pay a fee. They are called "takers". This is described at:
        https://docs.polymarket.com/market-makers/maker-rebates

## System architecture

There is some code, not in this repository, which is responsible for pulling data from the Polymarket. It pulls data every two seconds and records it in a JSON file, which is in the data folder. It breaks the data up into the five-minute increments and has some metadata about each of the periods. This is described in the reference document 2026-03-16-tsv-combiner.md. This document should only be used to understand the JSON file. It is not relevant to the construction of this system.

This repository will be responsible for creating the AI and training it on this data set. Then the eventual goal is to use the Polymarket API to do automated trades on that platform.

This code is hosted in a container. For training, the goal will be to move to a GPU-based cloud system to run the training. I do not know if that will be necessary for inference.

## AI Design

## Training

The methodology of the AI training will be similar to training it to play a game. Each episode of data will be fed to the AI one row at a time, along with SOME of the meta data at the top of the 5 min episode. The AI should not be fed information It isn't supposed to know yet. After the AI receives a line of input, it has essentially nine choices, in three categories.

    ### Do nothing. 
        This is a category in and of itself and is the only action in this category. 
    ### Fill a currently active order.
        - Buy one "up" share at the current ask price.
        - Sell one "up" share at the current bid price.
        - Buy one "down" share at the current ask price. 
        - Sell one "down" share at the current bid price.
    ### Enter an order in the order book. 
        - Enter a buy order for one "up" share at one penny below the current ask price.
        - Enter a sell order for one "up" share at one penny above the current bid price.
        - Enter a buy order for one "down" share at one penny below the current ask price. 
        - Enter a sell order for one "down" share at one penny above the current bid price.

        - Buys and sells should always be constrained to a range of between 1¢ and 99¢. Orders cannot be placed in the order book for higher or lower than these values. At times, a null will appear in the data for the current order book price. That means no orders can be filled or entered in the order book for that particular price and share type combination. 

        - Once an order is entered, it is considered filled if, in the future row during that epoch, the price matches the order. For example, if an AI entered a sell order for one up share at 4 cents, and there was a row after that episode where there was a bid for an up share at the price of 4 cents or higher, then the order would be considered filled. If the order is not filled by the end of the episode, the AI does not make or lose any money.

        - If, during decision time after the AI is exposed to a line of input, it decides either to fill a currently active order or enter an order in the order book, the AI will not do anything else for the remainder of the episode.

        - If the AI ended up selling a share either by filling a previously existing order or by having an order it puts on the order book, filled, then it immediately receives the amount of money it sold it for. Conversely, if it buys a share, it is losing the amount of money it bought the share for. 
        - At the end of the episode: The AI will receive $1 if it bought an up share when the Bitcoin price is above the price to beat price at the end of the episode, or conversely if it bought a down share, It will receive $1 if the episode ends with the price of Bitcoin being below the price to beat. If it bought the wrong share, it will gain nothing and be out the amount of money that it paid.
        - Conversely, at the end of the episode, the AI will lose a dollar if it sold an "up" share and the price to beat is below the price at the end of the episode. And the AI will lose a dollar if it sold a "down" share and the price to beat is above the price at the end of the episode.

        - The amount of money it makes or loses is used to train the AI as a reward function, proportional to the amount of money it either gained (which would, of course, be a positive reward) or the amount of money it lost (which, of course, would be a negative discouraging signal). After each episode, this reward should serve to do the training.

        ### Datafields
        - To make a decision, the AI has the following data fields at its disposal.
            - From the metadata for the episode:
                "hour",
                "day",
                "diff_pct_prev_session",
                "diff_pct_hour"
            - From the current row:
                "up_bid" ,
                "up_ask",
                "down_bid",
                "down_ask",
                "diff_pct",
                "time_to_close"


