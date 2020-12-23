"""
some utilities necessary for development
"""

import random


def demo_tile_transforms(deck):
    print("--- Demo Tile Transforms ---")
    tile = deck[0]

    print("Initial tile:")
    print(tile)

    print(">> flipped horizontally")
    print(tile.hflip())

    print(">> flipped horizontally again")
    print(tile.hflip())

    print(">> flipped vertically")
    print(tile.vflip())

    print(">> flipped vertically again")
    print(tile.vflip())

    print(">> flipped vertically 2 times")
    print(tile.vflip().vflip())

    print(">> rotate 2 steps")
    print(tile.rotate(6))

    print(">> rotate 1 step")
    print(tile.rotate(1))

    print(">> rotate 3 steps")
    print(tile.rotate(3))


def demo_fill_board_sequentially(board, deck):

    print("--- Filling the board sequentially --")
    while deck:
        places = board.free_places()
        print("Free spaces:", places)
        tile = deck[0]
        deck.take(tile)
        board.place(tile, places.pop(0))

    print("Free spaces:", board.free_places())


def demo_fill_board_randomly(board, deck):
    print("--- Demo Random Board Filling ---")

    places = board.places()
    random.shuffle(places)
    for pos, tile in zip(places, deck):
        board.place(tile, pos)

    print(board)


def demo_deck_capacities(deck):
    print("-- Demo deck functionality ---")
    print("Size:", len(deck))

    print("State of the deck:")
    reveal_deck(deck)

    tiles2remove = []
    for i in {1, 3}:
        print(f"Tile at [{i}] : {deck[i].id}")
        tiles2remove.append(deck[i])

    print("Taking (removing) the above tiles from the deck")
    for tile in tiles2remove:
        deck.take(tile)

    print("State of the deck:")
    reveal_deck(deck)

    print("Re-adding the tiles back to the deck")
    for tile in tiles2remove:
        deck.add(tile)

    print("State of the deck:")
    reveal_deck(deck)

    print("Removing and reading all tiles")
    tiles = list(deck.tiles)  # important to copy the list
    for tile in deck:
        deck.take(tile)

    print("State of the deck:")
    reveal_deck(deck)

    for tile in tiles:
        deck.add(tile)

    print("State of the deck:")
    reveal_deck(deck)

    id = 2311
    print(f"Searching the deck for a Tile by tile id: {id}")
    tile = deck.find(id)
    print(tile)
    print("Found?", tile.id == id)


def demo_tile_fitting(board, deck):

    tests = [
        ((1951, (0, 0)), (2311, (0, 1)),
         "should match side by side w/o transformations"),
        ((None, None),   (3079, (0, 2)),
         "Needs transforms"),
        ((None, None),   (2729, (1, 0)),
         "Needs transforms"),
        ((None, None),   (1427, (1, 1)),
         "Needs transforms"),
        ((None, None),   (2473, (1, 2)),
         "Needs transforms"),
        ((None, None),   (2971, (2, 0)),
         "Needs transforms"),
        ((None, None),   (1489, (2, 1)),
         "Needs transforms"),
        ((None, None),   (1171, (2, 2)),
         "Needs transforms"),
    ]

    tile_ids = [2311, 3079, 2729, 1427, 2971, 1489, 1171]

    for (id1, pos1), (id2, pos2), msg in tests:
        if id1:
            base = deck.find(id1)
            base.transform('v')

            base_fitter = TileFitter(board, pos1, base)
            for i, _ in enumerate(base_fitter):
                print(i)
            board.place(base, pos1)

            print(f"--- Base Tile {base.id} ---")
            print(base)

        tile = deck.find(id2)
        print(f"\n--- Target: {pos2}, fitting tile: id={tile.id}")
        print(tile)

        fitter = TileFitter(board, pos2, tile)
        for i, _ in enumerate(fitter):
            print(f"State of the tile after {i}th '{fitter.transform_}':")
            print(tile)
            board.place(tile, pos2)
            print("=== Placed !!!")
            break

        suggestions = fitter.select(tile_ids)
        print("Suggested Tiles:", suggestions)

    # print("--- Final Deck ---")
    # reveal_deck(deck)
    print("--- Final Board ---")
    print(board)


def reveal_deck(deck):
    tiles = []
    for idx, tile in enumerate(deck):
        tiles.append(tile.id)
    print(f"Size: {len(deck)}, Tiles: {tiles}")
