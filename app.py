import streamlit as st
import pandas as pd
import joblib
from datetime import date

from daily_slate import get_games, get_hr_hitters
from current_form import get_current_team_form
from prediction_tracker import save_prediction, score_predictions
from strikeout_model import predict_pitcher_strikeouts
from espn_odds import get_espn_odds
from manual_tracker import load_manual_tracker, save_manual_tracker

st.set_page_config(
    page_title="MLB Prediction App",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ MLB Prediction Dashboard")
st.caption("Daily MLB predictions, HR picks, strikeouts, predicted hits, parlays, and manual tracking")

model = joblib.load("mlb_model.pkl")
win_model = joblib.load("win_model.pkl")
win_model_columns = joblib.load("win_model_columns.pkl")

df = pd.read_csv("batter_features_named.csv")
pitcher_stats = pd.read_csv("pitcher_stats.csv")
team_stats = pd.read_csv("team_stats.csv")

st.sidebar.header("Controls")
selected_date = st.sidebar.date_input("Game Date", date.today())

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🏠 Daily Slate",
    "💣 HR Picks",
    "🔥 Best HR Picks Today",
    "🏆 Win Predictions",
    "📊 Batter Dashboard",
    "🎯 Pitcher Strikeout Predictions",
    "🎯 Predicted Hits",
    "🧾 Model Parlay",
    "✅ Manual Pick Tracker",
    "📈 Model Performance"
])


def build_game_input(game, current_form):
    away_team = game["away_team"]
    home_team = game["home_team"]
    away_pitcher = game["away_probable_pitcher"]
    home_pitcher = game["home_probable_pitcher"]

    away_form = current_form[current_form["team"] == away_team]["current_last_10_win_pct"]
    home_form = current_form[current_form["team"] == home_team]["current_last_10_win_pct"]

    away_last_10 = away_form.iloc[0] if not away_form.empty else 0.5
    home_last_10 = home_form.iloc[0] if not home_form.empty else 0.5

    game_input = pd.DataFrame(0.0, index=[0], columns=win_model_columns)

    if "home_field_advantage" in game_input.columns:
        game_input.loc[0, "home_field_advantage"] = 0.4

    for col in [
        f"away_team_{away_team}",
        f"home_team_{home_team}",
        f"away_probable_pitcher_{away_pitcher}",
        f"home_probable_pitcher_{home_pitcher}",
    ]:
        if col in game_input.columns:
            game_input.loc[0, col] = 1

    if "away_last_10_win_pct" in game_input.columns:
        game_input.loc[0, "away_last_10_win_pct"] = away_last_10 * 0.35

    if "home_last_10_win_pct" in game_input.columns:
        game_input.loc[0, "home_last_10_win_pct"] = home_last_10 * 0.35

    away_team_row = team_stats[team_stats["team"] == away_team]
    home_team_row = team_stats[team_stats["team"] == home_team]

    team_stat_cols = [
        "runs_per_game",
        "team_avg",
        "team_obp",
        "team_slg",
        "team_ops",
        "team_strikeouts",
        "team_era",
        "team_whip",
        "team_pitching_k9",
        "team_pitching_hr9",
    ]

    for col in team_stat_cols:
        away_col = f"away_{col}"
        home_col = f"home_{col}"

        if away_col in game_input.columns and not away_team_row.empty:
            game_input.loc[0, away_col] = away_team_row.iloc[0][col]

        if home_col in game_input.columns and not home_team_row.empty:
            game_input.loc[0, home_col] = home_team_row.iloc[0][col]

    return game_input, away_last_10, home_last_10


def predict_game(game, current_form):
    game_input, away_last_10, home_last_10 = build_game_input(game, current_form)

    home_win_prob = win_model.predict_proba(game_input)[0][1] * 100
    away_win_prob = 100 - home_win_prob

    away_team = game["away_team"]
    home_team = game["home_team"]

    predicted_winner = home_team if home_win_prob > away_win_prob else away_team
    confidence = max(away_win_prob, home_win_prob)

    return away_win_prob, home_win_prob, predicted_winner, confidence, away_last_10, home_last_10


with tab1:
    st.header("Daily MLB Games")

    games = get_games(selected_date.strftime("%Y-%m-%d"))

    if games.empty:
        st.warning("No games found for this date.")
    else:
        for _, row in games.iterrows():
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 1, 2])

                with col1:
                    st.subheader(row["away_team"])
                    st.caption(f"Pitcher: {row['away_probable_pitcher']}")

                with col2:
                    st.markdown("## @")

                with col3:
                    st.subheader(row["home_team"])
                    st.caption(f"Pitcher: {row['home_probable_pitcher']}")


with tab2:
    st.header("Likely Home Run Hitters")

    games = get_games(selected_date.strftime("%Y-%m-%d"))

    if games.empty:
        st.warning("No games found for this date.")
    else:
        game_options = games["away_team"] + " at " + games["home_team"]
        selected_game = st.selectbox("Choose a game", game_options)

        game_row = games[game_options == selected_game].iloc[0]

        away_team = game_row["away_team"]
        home_team = game_row["home_team"]

        st.info(
            f"{away_team} hitters face {game_row['home_probable_pitcher']} | "
            f"{home_team} hitters face {game_row['away_probable_pitcher']}"
        )

        hr_hitters = get_hr_hitters()

        game_hitters = hr_hitters[
            (hr_hitters["team"] == away_team) |
            (hr_hitters["team"] == home_team)
        ].head(10)

        if game_hitters.empty:
            st.warning("No hitter data found for this matchup.")
        else:
            for i, (_, row) in enumerate(game_hitters.iterrows(), start=1):
                team = row.get("team", "Unknown Team")

                if team == away_team:
                    opposing_pitcher = game_row["home_probable_pitcher"]
                elif team == home_team:
                    opposing_pitcher = game_row["away_probable_pitcher"]
                else:
                    opposing_pitcher = "Unknown"

                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.subheader(f"#{i} {row.get('player_name', 'Unknown')}")
                        st.caption(f"{team} vs {opposing_pitcher}")

                    with col2:
                        st.metric("HR Confidence", f"{row.get('hr_confidence', 0):.1f}%")

                    with col3:
                        st.metric("Avg EV", f"{row.get('avg_hit_speed', 0):.1f}")

                    with col4:
                        st.metric("Max EV", f"{row.get('max_hit_speed', 0):.1f}")

            with st.expander("View full HR hitter table"):
                display_cols = [
                    "player_name",
                    "team",
                    "hr_confidence",
                    "avg_hit_speed",
                    "max_hit_speed",
                    "brl_percent",
                    "hr_score"
                ]

                available_cols = [col for col in display_cols if col in game_hitters.columns]

                st.dataframe(game_hitters[available_cols], use_container_width=True)


with tab3:
    st.header("🔥 Best HR Picks Today")

    games = get_games(selected_date.strftime("%Y-%m-%d"))

    if games.empty:
        st.warning("No games found for this date.")
    else:
        hr_hitters = get_hr_hitters()

        today_teams = set(games["away_team"]).union(set(games["home_team"]))

        best_picks = hr_hitters[
            hr_hitters["team"].isin(today_teams)
        ].sort_values(
            "hr_confidence",
            ascending=False
        ).head(15)

        if best_picks.empty:
            st.warning("No HR picks found for today.")
        else:
            for i, (_, row) in enumerate(best_picks.iterrows(), start=1):
                player_team = row.get("team", "Unknown Team")

                matching_game = games[
                    (games["away_team"] == player_team) |
                    (games["home_team"] == player_team)
                ]

                if not matching_game.empty:
                    game = matching_game.iloc[0]

                    if player_team == game["away_team"]:
                        opposing_pitcher = game["home_probable_pitcher"]
                    else:
                        opposing_pitcher = game["away_probable_pitcher"]

                    matchup = f"{game['away_team']} at {game['home_team']}"
                else:
                    opposing_pitcher = "Unknown"
                    matchup = "Unknown matchup"

                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.subheader(f"#{i} {row.get('player_name', 'Unknown')}")
                        st.caption(f"{player_team} vs {opposing_pitcher}")
                        st.caption(matchup)

                    with col2:
                        st.metric("HR Confidence", f"{row.get('hr_confidence', 0):.1f}%")

                    with col3:
                        st.metric("Avg EV", f"{row.get('avg_hit_speed', 0):.1f}")

                    with col4:
                        st.metric("Barrel %", f"{row.get('brl_percent', 0):.1f}")

            with st.expander("View full best picks table"):
                display_cols = [
                    "player_name",
                    "team",
                    "hr_confidence",
                    "avg_hit_speed",
                    "max_hit_speed",
                    "brl_percent",
                    "hr_score"
                ]

                available_cols = [col for col in display_cols if col in best_picks.columns]

                st.dataframe(best_picks[available_cols], use_container_width=True)


with tab4:
    st.header("🏆 Win Predictions")

    games = get_games(selected_date.strftime("%Y-%m-%d"))

    if games.empty:
        st.warning("No games found for this date.")
    else:
        current_form = get_current_team_form()
        espn_odds = get_espn_odds()
        daily_manual_picks = []

        log_predictions = st.button("Save Today’s Predictions")
        save_manual_button = st.button("Save Today’s Model Picks to Manual Tracker")

        for _, game in games.iterrows():
            away_team = game["away_team"]
            home_team = game["home_team"]

            away_pitcher = game["away_probable_pitcher"]
            home_pitcher = game["home_probable_pitcher"]

            away_win_prob, home_win_prob, predicted_winner, confidence, away_last_10, home_last_10 = predict_game(
                game,
                current_form
            )

            daily_manual_picks.append({
                "game_date": selected_date.strftime("%Y-%m-%d"),
                "game": f"{away_team} at {home_team}",
                "pick": predicted_winner,
                "confidence": round(confidence, 1),
                "result": "Pending"
            })

            if log_predictions:
                save_prediction(
                    selected_date.strftime("%Y-%m-%d"),
                    away_team,
                    home_team,
                    predicted_winner,
                    away_win_prob,
                    home_win_prob
                )

            if not espn_odds.empty and "away_team" in espn_odds.columns and "home_team" in espn_odds.columns:
                odds_row = espn_odds[
                    (espn_odds["away_team"] == away_team) &
                    (espn_odds["home_team"] == home_team)
                ]
            else:
                odds_row = pd.DataFrame()

            with st.container(border=True):
                st.subheader(f"{away_team} at {home_team}")

                st.caption(
                    f"{away_team} pitcher: {away_pitcher} | "
                    f"{home_team} pitcher: {home_pitcher}"
                )

                if not odds_row.empty:
                    odds_details = odds_row.iloc[0].get("details", "No odds details")
                    over_under = odds_row.iloc[0].get("over_under", "N/A")

                    st.caption(f"ESPN Line: {odds_details}")
                    st.caption(f"Over/Under: {over_under}")
                else:
                    st.caption("ESPN Line: Not available")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(away_team, f"{away_win_prob:.1f}%")

                with col2:
                    st.metric("Predicted Winner", predicted_winner)

                with col3:
                    st.metric(home_team, f"{home_win_prob:.1f}%")

                with col4:
                    st.metric("Form Edge", f"{home_last_10 - away_last_10:+.2f}")

                st.progress(int(confidence))

                if predicted_winner == home_team:
                    reason = (
                        f"{home_team} is favored using team history, probable pitcher, "
                        f"team offense/pitching, home field, and recent form."
                    )
                else:
                    reason = (
                        f"{away_team} is favored using team history, probable pitcher, "
                        f"team offense/pitching, road matchup, and recent form."
                    )

                st.info(reason)

        if save_manual_button:
            tracker = load_manual_tracker()
            new_picks = pd.DataFrame(daily_manual_picks)

            tracker = pd.concat([tracker, new_picks], ignore_index=True)

            tracker = tracker.drop_duplicates(
                subset=["game_date", "game", "pick"],
                keep="last"
            )

            save_manual_tracker(tracker)

            st.success("Today’s model picks saved to Manual Pick Tracker.")

        if log_predictions:
            st.success("Today’s predictions saved to prediction_log.csv.")


with tab5:
    st.header("Batter Strength Predictor")

    player_name = st.selectbox("Choose Batter", df["player_name"])
    selected = df[df["player_name"] == player_name].iloc[0]

    st.subheader(player_name)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Exit Velocity", f"{selected['avg_exit_velocity']:.2f} mph")

    with col2:
        st.metric("Max Exit Velocity", f"{selected['max_exit_velocity']:.2f} mph")

    with col3:
        st.metric("Avg Launch Angle", f"{selected['avg_launch_angle']:.2f}°")

    with col4:
        st.metric("Batted Balls", int(selected["batted_balls"]))

    features = [[
        selected["avg_exit_velocity"],
        selected["max_exit_velocity"],
        selected["avg_launch_angle"],
        selected["batted_balls"]
    ]]

    if st.button("Predict Batter Strength"):
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            st.success(f"Strong hitter probability: {probability:.2%}")
        else:
            st.warning(f"Strong hitter probability: {probability:.2%}")

    st.divider()

    st.header("Top 20 Average Exit Velocity")

    leaderboard = df.sort_values("avg_exit_velocity", ascending=False).head(20)

    st.dataframe(
        leaderboard[[
            "player_name",
            "avg_exit_velocity",
            "max_exit_velocity",
            "avg_launch_angle",
            "batted_balls"
        ]],
        use_container_width=True
    )

    st.subheader("Exit Velocity Chart")

    chart_data = leaderboard.set_index("player_name")["avg_exit_velocity"]
    st.bar_chart(chart_data)


with tab6:
    st.header("🎯 Starting Pitcher Strikeout Predictions")

    games = get_games(selected_date.strftime("%Y-%m-%d"))

    if games.empty:
        st.warning("No games found for this date.")
    else:
        for _, game in games.iterrows():
            away_team = game["away_team"]
            home_team = game["home_team"]

            away_pitcher = game["away_probable_pitcher"]
            home_pitcher = game["home_probable_pitcher"]

            away_pitcher_id = game.get("away_probable_pitcher_id")
            home_pitcher_id = game.get("home_probable_pitcher_id")

            away_k = predict_pitcher_strikeouts(
                away_pitcher_id,
                opponent_team=home_team
            )

            home_k = predict_pitcher_strikeouts(
                home_pitcher_id,
                opponent_team=away_team
            )

            with st.container(border=True):
                st.subheader(f"{away_team} at {home_team}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"### {away_pitcher}")
                    st.caption(f"{away_team} starter")

                    if away_k:
                        st.metric("Expected Strikeouts", f"{away_k['expected_strikeouts']:.1f}")
                        st.caption(f"Avg K/start: {away_k['avg_k_per_start']:.1f}")
                        st.caption(f"Opponent K Factor: {away_k.get('opponent_k_factor', 1):.2f}")

                        if away_k.get("opponent_k_per_game") is not None:
                            st.caption(f"Opponent batter K/game: {away_k['opponent_k_per_game']:.1f}")

                        st.caption(f"ERA: {away_k['era']} | WHIP: {away_k['whip']}")
                    else:
                        st.warning("No pitcher data available")

                with col2:
                    st.markdown(f"### {home_pitcher}")
                    st.caption(f"{home_team} starter")

                    if home_k:
                        st.metric("Expected Strikeouts", f"{home_k['expected_strikeouts']:.1f}")
                        st.caption(f"Avg K/start: {home_k['avg_k_per_start']:.1f}")
                        st.caption(f"Opponent K Factor: {home_k.get('opponent_k_factor', 1):.2f}")

                        if home_k.get("opponent_k_per_game") is not None:
                            st.caption(f"Opponent batter K/game: {home_k['opponent_k_per_game']:.1f}")

                        st.caption(f"ERA: {home_k['era']} | WHIP: {home_k['whip']}")
                    else:
                        st.warning("No pitcher data available")


with tab7:
    st.header("🎯 Predicted Batter Hits")
    st.caption("Version 2 filters hitters by the selected game teams and ranks likely hit production.")

    games = get_games(selected_date.strftime("%Y-%m-%d"))

    if games.empty:
        st.warning("No games found for this date.")
    else:
        game_options = games["away_team"] + " at " + games["home_team"]

        selected_hit_game = st.selectbox(
            "Choose a game for hit predictions",
            game_options
        )

        game_row = games[game_options == selected_hit_game].iloc[0]

        away_team = game_row["away_team"]
        home_team = game_row["home_team"]

        st.subheader(selected_hit_game)

        hr_hitters = get_hr_hitters()

        game_players = hr_hitters[
            (hr_hitters["team"] == away_team) |
            (hr_hitters["team"] == home_team)
        ].copy()

        batter_df = df.copy()

        batter_df = batter_df.merge(
            game_players[["player_id", "team"]],
            left_on="batter",
            right_on="player_id",
            how="inner"
        )

        if batter_df.empty:
            st.warning("No batter data found for this matchup.")
        else:
            batter_df["predicted_hits"] = (
                0.45
                + (batter_df["avg_exit_velocity"] - 85) * 0.03
                + (batter_df["max_exit_velocity"] - 100) * 0.01
                + (batter_df["batted_balls"] / batter_df["batted_balls"].max()) * 0.35
                - abs(batter_df["avg_launch_angle"] - 12) * 0.01
            )

            batter_df["predicted_hits"] = batter_df["predicted_hits"].clip(
                lower=0.1,
                upper=2.5
            )

            batter_df["hit_confidence"] = (
                batter_df["predicted_hits"] /
                batter_df["predicted_hits"].max() *
                100
            )

            batter_df = batter_df.sort_values("predicted_hits", ascending=False)

            team_filter = st.selectbox(
                "Filter team",
                ["Both Teams", away_team, home_team]
            )

            if team_filter != "Both Teams":
                batter_df = batter_df[batter_df["team"] == team_filter]

            for i, (_, row) in enumerate(batter_df.head(12).iterrows(), start=1):
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.subheader(f"#{i} {row['player_name']}")
                        st.caption(row["team"])

                    with col2:
                        st.metric("Predicted Hits", f"{row['predicted_hits']:.2f}")

                    with col3:
                        st.metric("Hit Confidence", f"{row['hit_confidence']:.1f}%")

                    with col4:
                        st.metric("Avg EV", f"{row['avg_exit_velocity']:.1f}")

            with st.expander("View full predicted hits table"):
                st.dataframe(
                    batter_df[[
                        "player_name",
                        "team",
                        "predicted_hits",
                        "hit_confidence",
                        "avg_exit_velocity",
                        "max_exit_velocity",
                        "avg_launch_angle",
                        "batted_balls"
                    ]],
                    use_container_width=True
                )


with tab8:
    st.header("🧾 Model Parlay Builder")
    st.caption("Educational/statistical suggestions only. Parlays are high-risk.")

    games = get_games(selected_date.strftime("%Y-%m-%d"))

    if games.empty:
        st.warning("No games found for this date.")
    else:
        current_form = get_current_team_form()
        parlay_candidates = []

        for _, game in games.iterrows():
            away_win_prob, home_win_prob, predicted_winner, confidence, _, _ = predict_game(
                game,
                current_form
            )

            parlay_candidates.append({
                "game": f"{game['away_team']} at {game['home_team']}",
                "pick": predicted_winner,
                "confidence": confidence,
                "away_pitcher": game["away_probable_pitcher"],
                "home_pitcher": game["home_probable_pitcher"]
            })

        parlay_df = pd.DataFrame(parlay_candidates).sort_values(
            "confidence",
            ascending=False
        )

        min_confidence = st.slider(
            "Minimum confidence",
            min_value=50,
            max_value=90,
            value=55
        )

        legs = st.slider(
            "Number of parlay legs",
            min_value=5,
            max_value=7,
            value=5
        )

        parlay = parlay_df[
            parlay_df["confidence"] >= min_confidence
        ].head(legs)

        if parlay.empty:
            st.warning("No parlay picks meet that confidence level.")
        else:
            st.subheader("Suggested Model Parlay")

            for i, (_, row) in enumerate(parlay.iterrows(), start=1):
                with st.container(border=True):
                    st.markdown(f"### Leg {i}: {row['pick']}")
                    st.caption(row["game"])
                    st.metric("Model Confidence", f"{row['confidence']:.1f}%")
                    st.caption(
                        f"Pitching matchup: {row['away_pitcher']} vs {row['home_pitcher']}"
                    )

            combined_probability = (parlay["confidence"] / 100).prod() * 100

            st.metric(
                "Estimated Combined Hit Probability",
                f"{combined_probability:.1f}%"
            )

            st.warning(
                "Parlays are much harder to hit than single picks. Use this as a model card, not guaranteed betting advice."
            )

            with st.expander("View all parlay candidates"):
                st.dataframe(parlay_df, use_container_width=True)


with tab9:
    st.header("✅ Manual Pick Tracker")
    st.caption("Manually track model picks for each day.")

    tracker = load_manual_tracker()

    if tracker.empty:
        st.info("No picks saved yet. Go to Win Predictions and click Save Today’s Model Picks to Manual Tracker.")
    else:
        edited_tracker = st.data_editor(
            tracker,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "result": st.column_config.SelectboxColumn(
                    "Result",
                    options=["Pending", "Win", "Loss", "Push", "No Bet"],
                    required=False
                )
            }
        )

        if st.button("Save Manual Tracker"):
            save_manual_tracker(edited_tracker)
            st.success("Manual tracker saved.")

        completed = edited_tracker[
            edited_tracker["result"].isin(["Win", "Loss"])
        ]

        if not completed.empty:
            wins = (completed["result"] == "Win").sum()
            losses = (completed["result"] == "Loss").sum()
            accuracy = wins / len(completed) * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Tracked Picks", len(completed))

            with col2:
                st.metric("Wins", wins)

            with col3:
                st.metric("Manual Accuracy", f"{accuracy:.1f}%")


with tab10:
    st.header("📈 Model Performance")

    if st.button("Score Predictions / Update Performance"):
        try:
            score_predictions()
            st.success("Model performance updated.")
        except FileNotFoundError:
            st.warning("No prediction log found yet. Save predictions first.")
        except Exception as e:
            st.error(f"Could not score predictions: {e}")

    try:
        perf = pd.read_csv("prediction_log_scored.csv")

        total_predictions = len(perf)
        correct_predictions = perf["correct"].sum()
        accuracy = correct_predictions / total_predictions * 100 if total_predictions else 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Predictions", total_predictions)

        with col2:
            st.metric("Correct Picks", int(correct_predictions))

        with col3:
            st.metric("Accuracy", f"{accuracy:.1f}%")

        if "home_win_prob" in perf.columns:
            perf["confidence"] = perf[
                ["away_win_prob", "home_win_prob"]
            ].max(axis=1)

            bins = [0, 55, 60, 70, 80, 100]
            labels = ["50-55%", "55-60%", "60-70%", "70-80%", "80%+"]

            perf["confidence_bucket"] = pd.cut(
                perf["confidence"],
                bins=bins,
                labels=labels
            )

            bucket_accuracy = (
                perf.groupby("confidence_bucket")["correct"].mean()
                * 100
            )

            st.subheader("Confidence Accuracy")
            st.bar_chart(bucket_accuracy)

        st.subheader("Recent Predictions")

        st.dataframe(
            perf.tail(25),
            use_container_width=True
        )

    except FileNotFoundError:
        st.warning(
            "No scored prediction file found yet. Run scoring after games are final."
        )