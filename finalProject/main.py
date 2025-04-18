from analysis import stats
from analysis import parser
from analysis import matrix

def main():
    # df = parser.load_log('PokerLogs/24_9_19-log.csv')
    # player_dict = parser.create_player_dict(df)

    # #total calls, raises, bets, and folds
    # totalCalls = stats.get_action_counts(df, 'calls', player_dict)
    # totalFoldss = stats.get_action_counts(df, 'folds', player_dict, min_words=3)
    # totalRaises = stats.get_action_counts(df, 'raises', player_dict)
    # totalBets = stats.get_action_counts(df, 'bets', player_dict)

    # #agression factor, bets + raises divided by calls
    # agressionFactor = stats.calc_aggression_factor(totalBets, totalRaises, totalCalls, player_dict)

    # #pfr percent, indicator of agression
    # preflopRaisePercentage = stats.calc_PFR(df, player_dict)

    # #preflopstats
    # pfCalls = stats.get_preflop_actions(df, player_dict, 'calls')
    # pfRaises = stats.get_preflop_actions(df, player_dict, 'raises')
    # pfFolds = stats.get_preflop_actions(df, player_dict, 'folds')

    # #vpip, measure of how many hands a player participates in. (voluntary participation in pot)
    # vpip = stats.calc_VPIP(df, player_dict)
    
    # #total number of hands played
    # hands = stats.track_player_presence(df, player_dict)


    # #P/L CALCULATIONS, THE STRUCTURE OF THE LOG MAKES THESE COMPLICATED

    # #profitstats
    # takenProfit = stats.get_quit_or_stand_stacks_all(df, player_dict)
    # remainingProfit = stats.get_final_player_stacks(df, player_dict)
    # extraProfit = stats.get_quit_or_stand_stacks_after_final(df, player_dict)
    # grossProfit = stats.calculate_gross_profits(player_dict, takenProfit, remainingProfit, extraProfit)

    
    # #buyin stats
    # joinedBuyIn = stats.get_joined_buy_ins(df, player_dict)
    # adminUpdatedBuyIn = stats.get_admin_updates_after_game_start(df, player_dict)
    # trueBuyIn = stats.calculate_final_buyin(player_dict, joinedBuyIn, adminUpdatedBuyIn)

    # netProfit = stats.calculate_net_profit(player_dict, grossProfit, trueBuyIn)
    # profitYesOrNo = stats.profit_classifier(player_dict, netProfit)

    
    # # print(grossProfit)
    # # print(trueBuyIn)
    # # print(netProfit)
    # # print(profitYesOrNo)

    # showCounter = stats.count_shows(df, player_dict)
    # standCounter= stats.count_stands(df, player_dict)
    # print("Number of Shows: ", showCounter)
    # print("Number of times standing up: ", standCounter)
    # print("Preflop Raise Percentage: ", preflopRaisePercentage)
    # print("Number of hands", hands)




    profitMatrix, yesOrNoMatrix  = matrix.constructMatrix()
    profitMatrix.to_csv('profitAmountCSVs/player_stats.csv', index = False)

    yesOrNoMatrix.to_csv('profitYesOrNoCSVs/player_stats.csv', index = False)

    



if __name__ == '__main__': main()