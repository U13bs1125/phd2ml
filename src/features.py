def get_feature_sets(feature_dict):
    return {
        "weather": feature_dict["weather"],
        "weather_soil": feature_dict["weather"] + feature_dict["soil"],
        "weather_soil_agro": feature_dict["weather"] + feature_dict["soil"] + feature_dict["agro"]
    }