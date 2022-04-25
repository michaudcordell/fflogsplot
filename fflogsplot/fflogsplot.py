# Standard library imports
import json
import os
import pathlib as pl
from typing import Dict

# Third-party imports
import dotenv
import gql
import gql.transport.aiohttp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.linear_model


def extract_data():
    dotenv.load_dotenv()

    fight_prog_query_path = pl.Path("queries") / "fight_prog_query.graphql"
    with open(fight_prog_query_path, "r") as fight_prog_query_file:
        fight_prog_query = gql.gql(fight_prog_query_file.read())

    fight_prog_query_variables = {
        "userID": os.environ["USER_ID"],
        "encounterID": os.environ["ENCOUNTER_ID"]
    }
    fight_prog_query_op_name = "GetFightProgDataByEncounterID"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['ACCESS_TOKEN']}"
    }
    dest_url = "https://www.fflogs.com/api/v2/user"
    transport = gql.transport.aiohttp.AIOHTTPTransport(
        url=dest_url, headers=headers)
    client = gql.Client(transport=transport)
    response = client.execute(fight_prog_query, 
                            variable_values=fight_prog_query_variables, 
                            operation_name=fight_prog_query_op_name)["reportData"]
    response["reports"]["data"] = [reportdata for reportdata in response["reports"]["data"] 
                                   if reportdata["title"].startswith("ucob day ")]

    json_data_path = pl.Path("data") / "fight_prog.json"
    with open(json_data_path, "w") as json_data_file:
        json_data_file.write(json.dumps(response, indent=4))

def process_data(raw_json_dict: Dict):
    processed_fight_prog_data = {
        "Cumulative Pulls": [],
        "Day": [],
        "Pull": [],
        "Pull Start Time": [],
        "Pull End Time": [],
        "Pull Phase": [],
        "Historical Highest Phase": [],
        "Pull Fight Completion": [],
        "Historical Highest Fight Completion": [],
        "Pull Duration": [],
        "Daily Cumulative Hours": [],
        "Total Cumulative Hours": []
    }
    
    fight_prog_data_raw = [
        reportdict for reportdict in raw_json_dict["reports"]["data"][::-1]]

    # Iterate through each report
    cumulative_pulls = 1
    total_cumulative_hours = 0.0
    highest_phase = 0
    highest_fight_completion = 0.0
    for reportdict in fight_prog_data_raw:
        day = int(reportdict["title"].split()[-1])
        daily_cumulative_hours = 0.0
        
        # Iterate through each pull in the report
        for pull_index, fightdict in enumerate(reportdict["fights"]):
            pull_duration = (
                float(fightdict["endTime"]) - float(fightdict["startTime"])) / 1000
            daily_cumulative_hours += pull_duration / 60 / 60
            total_cumulative_hours += pull_duration / 60 / 60
            highest_phase = max(highest_phase, 
                                int(fightdict["lastPhaseAsAbsoluteIndex"]))
            highest_fight_completion = max(highest_fight_completion, 
                                           100 - float(fightdict["fightPercentage"]))
            
            processed_fight_prog_data["Cumulative Pulls"].append(
                cumulative_pulls)
            processed_fight_prog_data["Day"].append(day)
            processed_fight_prog_data["Pull"].append(pull_index + 1)
            processed_fight_prog_data["Pull Start Time"].append(
                float(fightdict["startTime"]))
            processed_fight_prog_data["Pull End Time"].append(
                float(fightdict["endTime"]))
            processed_fight_prog_data["Pull Phase"].append(
                int(fightdict["lastPhaseAsAbsoluteIndex"]))
            processed_fight_prog_data["Historical Highest Phase"].append(
                highest_phase)
            processed_fight_prog_data["Pull Fight Completion"].append(
                100 - float(fightdict["fightPercentage"]))
            processed_fight_prog_data["Historical Highest Fight Completion"].append(
                highest_fight_completion)
            processed_fight_prog_data["Pull Duration"].append(pull_duration)
            processed_fight_prog_data["Daily Cumulative Hours"].append(
                daily_cumulative_hours)
            processed_fight_prog_data["Total Cumulative Hours"].append(
                total_cumulative_hours)
            
            cumulative_pulls += 1
    
    processed_fight_prog_data_df = pd.DataFrame.from_dict(processed_fight_prog_data)
    
    processed_daily_fight_prog_data_df = \
        processed_fight_prog_data_df.groupby("Day")["Pull Fight Completion"].mean().reset_index().rename(
            columns={"Pull Fight Completion": "Mean Daily Fight Completion"})
    processed_daily_fight_prog_data_df["Median Daily Fight Completion"] = \
        processed_fight_prog_data_df.groupby("Day")["Pull Fight Completion"].median().reset_index().rename(
            columns={"Pull Fight Completion": "Median Daily Fight Completion"})["Median Daily Fight Completion"]

    highest_cumulative_hours_by_day_df = \
        processed_fight_prog_data_df.groupby("Day")["Daily Cumulative Hours"].max().reset_index()
    
    processed_fight_prog_data_df["Highest Daily Cumulative Hours"] = \
        processed_fight_prog_data_df.apply(
            lambda row: highest_cumulative_hours_by_day_df.loc[highest_cumulative_hours_by_day_df["Day"] == row["Day"], 
                                                               "Daily Cumulative Hours"].item(), axis=1)

    processed_fight_prog_data_path = pl.Path("data") / "fight_prog_processed.csv"
    processed_fight_prog_data_df.to_csv(processed_fight_prog_data_path)
    
    processed_daily_fight_prog_data_path = pl.Path("data") / "daily_fight_prog_processed.csv"
    processed_daily_fight_prog_data_df.to_csv(processed_daily_fight_prog_data_path)

def plot_data():
    # Load dataset
    ucobdata_path = pl.Path("data") / "fight_prog_processed.csv"
    ucobdata = pd.read_csv(ucobdata_path)
    
    ucobdata_daily_path = pl.Path("data") / "daily_fight_prog_processed.csv"
    ucobdata_daily = pd.read_csv(ucobdata_daily_path)
    
    ucobdata_daily_diff_path = pl.Path("data") / "daily_fight_prog_diff.csv"
    ucobdata_daily_diff = pd.read_csv(ucobdata_daily_diff_path)
    
    results_dir = pl.Path("results")
    
    # Plot Fight Completion over Cumulative Pulls
    sns.set_theme(style="darkgrid", palette="flare")
    fig, ax = plt.subplots()
    sns.scatterplot(data=ucobdata, x="Cumulative Pulls", y="Pull Fight Completion")
    plt.title("Fight Progress by Pull")
    ax.set_ylabel("Fight Progress (%)")
    ax.set_ylim(0, 100)
    plt.savefig(results_dir / "fight_completion_cumulative_pull_scatter.png", dpi=300)

    # Plot Median Daily Fight Progress
    sns.set_theme(style="darkgrid", palette="flare")
    fig, ax = plt.subplots()
    ucobdata_daily.rename(columns={"Mean Daily Fight Completion": "Mean", 
                                   "Median Daily Fight Completion": "Median"}, 
                          inplace=True)
    daily_fight_prog_data_melted = ucobdata_daily.melt(
        id_vars=["Day"], value_vars=["Mean", "Median"], var_name="Measure", 
        value_name="Daily Fight Completion")
    sns.lineplot(x="Day", y="Daily Fight Completion", hue="Measure", 
                 style="Measure", data=daily_fight_prog_data_melted)
    plt.title("Daily Fight Progress Measures")
    ax.set_ylabel("Fight Progress (%)")
    ax.set_ylim(0, 100)
    plt.savefig(results_dir / "fight_completion_measure_line.png", dpi=300)
    
    # Plot Highest Historical Fight Completion
    fig, ax = plt.subplots()
    sns.set_theme(style="darkgrid", palette="flare")
    sns.lineplot(x="Day", y="Historical Highest Fight Completion", ci=None, 
                 estimator=lambda vals: max(vals), 
                 data=ucobdata[["Day", "Historical Highest Fight Completion"]])
    plt.title("Highest Historical Fight Progress by Day")
    ax.set_ylabel("Fight Progress (%)")
    ax.set_ylim(0, 100)
    plt.savefig(results_dir / "historical_fight_completion_line.png", dpi=300)
    
    # Plot box plots of Fight Progress by Day
    fig, ax = plt.subplots()
    sns.set_theme(style="darkgrid", palette="flare")
    sns.boxplot(x="Day", y="Pull Fight Completion", 
                data=ucobdata)
    plt.title("Daily Fight Progress")
    ax.set_ylabel("Fight Progress (%)")
    ax.set_ylim(0, 100)
    plt.savefig(results_dir / "fight_completion_boxes.png", dpi=300)
    
    # Plot scatterplot and regression line by Day
    days = ucobdata_daily["Day"].to_numpy().reshape(-1, 1)
    fight_progs = ucobdata_daily["Median"].to_numpy()
    regression_model = sklearn.linear_model.LinearRegression(n_jobs=-1).fit(days, fight_progs)
    fig, ax = plt.subplots()
    sns.set_theme(style="darkgrid", palette="pastel")
    sns.scatterplot(x="Day", y="Median",
                    data=ucobdata_daily)
    plt.title(f"Daily Fight Progress wth Extrapolation (R^2 = {regression_model.score(days, fight_progs):.3f})")
    ax.set_ylabel("Fight Progress (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, (100 - regression_model.intercept_) / regression_model.coef_)
    plt.plot([0, (100 - regression_model.intercept_) / regression_model.coef_], [regression_model.intercept_, 100], "k--")
    plt.savefig(results_dir / "fight_completion_extrapolated.png", dpi=300)

def main():
    extract_data()
    
    fight_prog_data_path = pl.Path("data") / "fight_prog.json"
    with open(fight_prog_data_path, "r") as fight_prog_data_file:
        process_data(json.loads(fight_prog_data_file))
    
    plot_data()

if __name__ == "__main__":
    main()
