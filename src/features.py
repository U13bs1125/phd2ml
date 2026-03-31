def get_feature_sets(feature_dict):
    return {
        "weather": feature_dict["weather"],
        "plus_soil": feature_dict["weather"] + feature_dict["soil"],
        "plus_agro": feature_dict["weather"] + feature_dict["soil"] + feature_dict["agro"]
    }