#stats.py

from collections import defaultdict
import re
import math


def get_action_counts(df, action, player_dict, min_words=5, amount_index=None):

    """calculates the total number of specific actions taken by each player through the whole game.
    E.g. call funciton with 'calls' to get the total number of calls. Returns a dictionary with player names: action counts"""

    result = {player_id: 0 for player_id in player_dict}
    
    for entry in df['entry']:
        if action in entry and len(entry.split()) >= min_words:
            player_id = entry.split()[2][:-1]
            if player_id in result:
                if amount_index is not None:
                    try:
                        amount = float(entry.split()[amount_index])
                        result[player_id] += round(amount, 2)
                    except ValueError:
                        continue
                else:
                    result[player_id] += 1

    #make the dict {player_name: action count} rather than {playerID: action count}
    name_result = {player_dict[player_id]: count for player_id, count in result.items()}
    return name_result

def track_player_presence(df, player_dict):
    """
    Returns a dictionary mapping player names to the number of hands they were present for.
    """
    hand_counts = defaultdict(int)
    current_players = set()

    for entry in df['entry']:
        entry = entry.strip()

        if entry.startswith("Player stacks:"):
            current_players = set()
            matches = re.findall(r'"([^"]+ @ [^"]+)"', entry)
            for match in matches:
                name, pid = match.split(" @ ")
                current_players.add(pid)

        elif entry.startswith("-- ending hand") and current_players:
            for pid in current_players:
                name = player_dict.get(pid, pid)  # fallback to ID if name missing
                hand_counts[name] += 1
            current_players = set()  # reset for next hand

    #we have to add 1 to avoid missing last hand
    hand_counts = {key: value + 1 for key, value in hand_counts.items()}

    return dict(hand_counts)



def get_preflop_actions(df, player_dict, target_action):
    """
    similar to get_action_counts except that it only counts actions that occur BEFORE the flop.
    Note that an action can be counted at most once per hand. E.g. two raises preflop still only counts as 1. This is a convention in poker stats.
    (preflop actions are generally considered very important in poker analysis)
    """

    action_counts = defaultdict(set)
    current_hand_id = None
    preflop_actions = []
    has_flop = False
    players_already_counted = set()

    for entry in reversed(df['entry']):
        entry = entry.strip()

        if entry.startswith("-- starting hand"):
            if current_hand_id is not None:
                for line in preflop_actions:
                    if target_action in line:
                        match = re.search(r'"[^"]+ @ ([^"]+)"', line)
                        if match:
                            pid = match.group(1)
                            if pid not in players_already_counted:
                                action_counts[pid].add(current_hand_id)
                                players_already_counted.add(pid)

            # Start new hand
            match = re.search(r'#(\d+)', entry)
            current_hand_id = int(match.group(1)) if match else None
            preflop_actions = []
            has_flop = False
            players_already_counted.clear()

        elif entry.startswith("-- ending hand"):
            continue  # Skip

        elif "Flop:" in entry:
            has_flop = True

        elif current_hand_id is not None and not has_flop:
            preflop_actions.append(entry)

    # Return final counts
    return {
        player_dict.get(pid, pid): len(hand_ids)
        for pid, hand_ids in action_counts.items()
    }


def calc_VPIP(df, player_dict):
    """
    Calculates VPIP (Voluntarily Put Money In Pot) for each player.
    VPIP = (# of times player called or raised preflop) / (total hands played)
    Returns a dictionary of player names and their VPIP as a percentage.
    """
    #get total hands played
    hands_played = track_player_presence(df, player_dict)

    #get hands where a player got involved preflop
    preflop_calls = get_preflop_actions(df, player_dict, 'calls')
    preflop_raises = get_preflop_actions(df, player_dict, 'raises')

    vpip = {}
    for player in hands_played:
        total_hands = hands_played[player]
        calls = preflop_calls.get(player, 0)
        raises = preflop_raises.get(player, 0)
        vpip_count = calls + raises



        vpip[player] = round((vpip_count / total_hands) * 100, 2) if total_hands > 0 else 0

        

    return vpip

def calc_PFR(df, player_dict):
    """Calculates preflop raise percentage for each player.
    number of raises preflop divided by total number of hands played"""
    hands_played = track_player_presence(df, player_dict)
    preflop_raises = get_preflop_actions(df, player_dict, 'raises')

    pfr = {}
    for player in hands_played:
        total_hands = hands_played[player]
        raises = preflop_raises.get(player, 0)
        pfr[player] = round(raises / total_hands*100, 2) if total_hands > 0 else 0

    return pfr


def calc_aggression_factor(bets, raises, calls, player_dict):
    """calculates agression factor, the ratio of a player's total bets + raises divided by their calls
    higher values indicate more aggressive players"""
    factor = {}
    for player_id in bets:
        agg = bets.get(player_id, 0) + raises.get(player_id, 0)
        calls_ = calls.get(player_id, 0)
        player_name = player_dict.get(player_id, player_id)  # fallback to ID if name missing
        factor[player_name] = round(agg / calls_, 2) if calls_ else float(0.2)
    return factor


#calculate all GrossProfit and loss up to the last hand with buyin: joined the game with, admin updated, sit back, p/l: quits, stand up


def get_joined_buy_ins(df, player_dict):
    """
    Calculates the total buy-in amount for each player based on game log entries.
    Returns a dictionary {player_name: total_buy_in_amount}.
    """
    buy_ins = defaultdict(float)

    for entry in df.itertuples(index=False):
        entry_str = entry.entry.strip()

        # Case 1: Player joined the game (normal buy-in)
        if "joined the game with a stack of" in entry_str:
            match = re.search(r'"([^"]+ @ [^"]+)" joined the game with a stack of (\d+\.\d+)', entry_str)
            if match:
                name_id = match.group(1)
                amount = float(match.group(2))
                player_name = player_dict.get(name_id.split(" @ ")[1], name_id.split(" @ ")[0])
                buy_ins[player_name] += round(amount, 2)


    return dict(buy_ins)





def get_quit_or_stand_stacks_after_final(df, player_dict):
    """
    Sums up stacks for players who quit or stand up,
    but ignores any that happen AFTER the first 'Player stacks:' line (game end).
    """
    player_stacks = defaultdict(float)
    game_end_found = False

    for entry in df.itertuples(index=False):
        entry_str = entry.entry.strip()

        # Detect the first 'Player stacks:' line (marks the end of the game)
        if "Player stacks:" in entry_str:
            game_end_found = True

        # Only process quit/stand-up actions AFTER the game end
        if not game_end_found:
            if "quits the game with a stack of" in entry_str:
                match = re.search(r'"([^"]+ @ [^"]+)" quits the game with a stack of (\d+\.\d+)', entry_str)
                if match:
                    name_id = match.group(1)
                    stack = float(match.group(2))
                    player_name = player_dict.get(name_id.split(" @ ")[1], name_id.split(" @ ")[0])
                    player_stacks[player_name] += round(stack, 2)

            elif "stand up with the stack of" in entry_str:
                match = re.search(r'"([^"]+ @ [^"]+)" stand up with the stack of (\d+\.\d+)', entry_str)
                if match:
                    name_id = match.group(1)
                    stack = float(match.group(2))
                    player_name = player_dict.get(name_id.split(" @ ")[1], name_id.split(" @ ")[0])
                    player_stacks[player_name] += round(stack,2)

    return dict(player_stacks)




def get_quit_or_stand_stacks_all(df, player_dict):
    """
    Processes the poker log in reverse order (because log is reverse chronological),
    capturing only quit/stand-up actions before the final Player stacks snapshot.
    """
    player_stacks = defaultdict(float)

    # Convert to a list first so it can be reversed
    entries = list(df.itertuples(index=False))

    for entry in entries:
        entry_str = entry.entry.strip()

        if "quits the game with a stack of" in entry_str:
            match = re.search(r'"([^"]+ @ [^"]+)" quits the game with a stack of (\d+\.\d+)', entry_str)
            if match:
                name_id = match.group(1)
                stack = float(match.group(2))
                player_name = player_dict.get(name_id.split(" @ ")[1], name_id.split(" @ ")[0])
                player_stacks[player_name] += round(stack,2)

        elif "stand up with the stack of" in entry_str:
            match = re.search(r'"([^"]+ @ [^"]+)" stand up with the stack of (\d+\.\d+)', entry_str)
            if match:
                name_id = match.group(1)
                stack = float(match.group(2))
                player_name = player_dict.get(name_id.split(" @ ")[1], name_id.split(" @ ")[0])
                player_stacks[player_name] += round(stack,2)

    return dict(player_stacks)



def get_final_player_stacks(df, player_dict):
    """
    Extracts the first 'Player stacks:' line from the log and returns a dictionary
    of player names and their associated stack amounts.
    """
    player_stacks = {}

    for entry in df.itertuples(index=False):
        entry_str = entry.entry.strip()

        # Find the first 'Player stacks:' line
        if "Player stacks:" in entry_str:
            # Parse the player stacks from the line
            matches = re.findall(r'"([^"]+ @ [^"]+)" \((\d+\.\d+)\)', entry_str)
            for match in matches:
                player_name_id = match[0]  # e.g. "Cal @ E18ViE79KI"
                stack = float(match[1])    # e.g. 86.19
                player_name = player_dict.get(player_name_id.split(" @ ")[1], player_name_id.split(" @ ")[0])  # Map player ID to name
                player_stacks[player_name] = round(stack,2)
            break  # Only the first occurrence is needed

    return player_stacks





#similar to getting buyouts at the end of the game, there are complications with getting buy ins at the beginning
#we have this function to get all the admin changes to stacks before the game starts
#we add this to buy ins separatley
def get_admin_updates_after_game_start(df, player_dict):
    """
    Looks for admin updates (stack changes) that occur after the first starting hand.
    Returns a dict of player: total stack added.
    """
    stack_updates = defaultdict(float)
    game_started = False

    #reverse the log
    for entry in df[::-1].itertuples(index=False): 
        entry_str = entry.entry.strip()

        # Once we find the 'starting hand #1' line, set game_started to True
        if "-- starting hand #1" in entry_str:
            game_started = True
            continue


        if game_started:

            if "updated the player" in entry_str and "stack from" in entry_str:
                match = re.search(r'updated the player "([^"]+ @ [^"]+)" stack from (\d+\.\d+) to (\d+\.\d+)', entry_str)
                if match:
                    name_id = match.group(1)
                    old_stack = float(match.group(2))
                    new_stack = float(match.group(3))
                    player_name = player_dict.get(name_id.split(" @ ")[1], name_id.split(" @ ")[0])

                    stack_difference = new_stack - old_stack
                    if stack_difference > 0:
                        stack_updates[player_name] += round(stack_difference,2)

    return dict(stack_updates)



def calculate_final_buyin(player_dict, joined, updated):
    final_buyin = {name: 0.0 for name in player_dict.values()}

    for player, amount in joined.items():
        if player in final_buyin:
            final_buyin[player] += round(amount,2)

    for player, amount in updated.items():
        if player in final_buyin:
            final_buyin[player] += round(amount,2)

    return {player: round(amount, 2) for player, amount in final_buyin.items()}

def calculate_gross_profits(player_dict, takenGrossProfit, remainingStacks, extraGrossProfit):

    final_profits = {name: 0.0 for name in player_dict.values()}
    

    for player, amount in takenGrossProfit.items():
        if player in final_profits:
            final_profits[player] += round(amount,2)
    

    for player, amount in remainingStacks.items():
        if player in final_profits:
            final_profits[player] += round(amount,2)

    for player, amount in extraGrossProfit.items():
        if player in final_profits:
            final_profits[player] -= round(amount,2)

    return {player: round(amount, 2) for player, amount in final_profits.items()}

def calculate_net_profit(playerDict, grossProfit, buyIn):
    net_profits = {name: 0.0 for name in playerDict.values()}

    for player, amount in grossProfit.items():
        if player in net_profits:
            net_profits[player] += round(amount,2)

    for player, amount in buyIn.items():
        net_profits[player] -= round(amount,2)


    return {player: round(amount, 2) for player, amount in net_profits.items()}


def profit_classifier(playerDict, netProfit):
    profitClassifier = {name: 0 for name in playerDict.values()}

    for player, amount in netProfit.items():
        if player in profitClassifier:
            if amount > 0:
                profitClassifier[player] = 1
                #some logs are corrupted and give a net profit of 0. We set it to -1 here so we can delete these rows later
            elif amount == 0:
                profitClassifier[player] = -1

    return profitClassifier


def count_shows(df, player_dict):
    """Counts how many times each player shows their cards (voluntarily or at showdown)."""
    result = {player_name: 0 for player_name in player_dict.values()}

    for entry in df.itertuples(index=False):
        entry_str = entry.entry.strip()

        if "shows" in entry_str:
            match = re.search(r'"([^"]+ @ [^"]+)" shows', entry_str)
            if match:
                player_name_id = match.group(1)  # e.g. "DLA @ tH-zqsM1Dh"
                name, player_id = player_name_id.split(" @ ")
                player_name = player_dict.get(player_id, name)  # fallback to name if ID missing
                if player_name in result:
                    result[player_name] += 1

    return result


def count_stands(df, player_dict):
    """Counts how many times each steps away from the game. I.e. they leave, but not because they ran out of money"""
    result = {player_name: 0 for player_name in player_dict.values()}

    for entry in df.itertuples(index=False):
        entry_str = entry.entry.strip()

        if "stand" in entry_str:
            match = re.search(r'"([^"]+ @ [^"]+)" stand', entry_str)
            if match:
                player_name_id = match.group(1)  # e.g. "DLA @ tH-zqsM1Dh"
                name, player_id = player_name_id.split(" @ ")
                player_name = player_dict.get(player_id, name)  # fallback to name if ID missing
                if player_name in result:
                    result[player_name] += 1

    return result





