# frozen_string_literal: true

require 'rails_helper'

RSpec.describe 'Api::V1::Games', type: :request do
  let(:sport) { create(:sport, :basketball) }
  let(:league) { create(:league, :ncaam, sport: sport) }
  let(:season) { create(:season, league: league) }
  let(:home_team) { create(:team, :duke) }
  let(:away_team) { create(:team, :unc) }

  before do
    league.teams << [home_team, away_team]
  end

  describe 'GET /api/v1/games' do
    it 'returns empty array when no games exist' do
      get '/api/v1/games'

      expect(response).to have_http_status(:ok)
      expect(json_response[:games]).to eq([])
      expect(json_response[:meta][:count]).to eq(0)
    end

    it 'returns games' do
      game = create(:game, league: league, season: season, home_team: home_team, away_team: away_team)

      get '/api/v1/games'

      expect(response).to have_http_status(:ok)
      expect(json_response[:games].length).to eq(1)
      expect(json_response[:games].first[:home_team][:code]).to eq('DUKE')
      expect(json_response[:games].first[:away_team][:code]).to eq('UNC')
    end

    it 'filters by league' do
      game = create(:game, league: league, season: season, home_team: home_team, away_team: away_team)
      
      other_sport = create(:sport, :baseball)
      other_league = create(:league, :mlb, sport: other_sport)
      other_season = create(:season, league: other_league)
      other_home = create(:team)
      other_away = create(:team)
      other_league.teams << [other_home, other_away]
      create(:game, league: other_league, season: other_season, home_team: other_home, away_team: other_away)

      get '/api/v1/games', params: { league: 'ncaam' }

      expect(response).to have_http_status(:ok)
      expect(json_response[:games].length).to eq(1)
      expect(json_response[:meta][:filters][:league]).to eq('ncaam')
    end

    it 'filters by date' do
      today_game = create(:game, league: league, season: season, home_team: home_team, away_team: away_team, start_time: Time.current.middle_of_day)
      
      other_home = create(:team)
      other_away = create(:team)
      league.teams << [other_home, other_away]
      tomorrow_game = create(:game, league: league, season: season, home_team: other_home, away_team: other_away, start_time: 1.day.from_now.middle_of_day)

      get '/api/v1/games', params: { date: Date.current.to_s }

      expect(response).to have_http_status(:ok)
      expect(json_response[:games].length).to eq(1)
    end
  end

  describe 'GET /api/v1/games/:id' do
    it 'returns a game with details' do
      game = create(:game, league: league, season: season, home_team: home_team, away_team: away_team)

      get "/api/v1/games/#{game.id}"

      expect(response).to have_http_status(:ok)
      expect(json_response[:game][:id]).to eq(game.id)
      expect(json_response[:game][:home_team][:code]).to eq('DUKE')
      expect(json_response[:game][:odds]).to be_nil
      expect(json_response[:game][:predictions]).to eq({})
    end

    it 'returns 404 for unknown game' do
      get '/api/v1/games/00000000-0000-0000-0000-000000000000'

      expect(response).to have_http_status(:not_found)
    end

    it 'includes odds when present' do
      game = create(:game, league: league, season: season, home_team: home_team, away_team: away_team)
      data_source = create(:data_source, :espn)
      create(:game_odds, game: game, data_source: data_source, spread_favorite_team: home_team, spread_value: -5.5, total_line: 145.5)

      get "/api/v1/games/#{game.id}"

      expect(response).to have_http_status(:ok)
      expect(json_response[:game][:odds][:spread][:favorite]).to eq('DUKE')
      expect(json_response[:game][:odds][:spread][:value]).to eq(-5.5)
      expect(json_response[:game][:odds][:total][:line]).to eq(145.5)
    end

    it 'includes predictions when present' do
      game = create(:game, league: league, season: season, home_team: home_team, away_team: away_team)
      data_source = create(:data_source, :gdo)
      create(:game_prediction, game: game, data_source: data_source, home_predicted_score: 75.5, away_predicted_score: 70.2, predicted_winner: home_team, model_version: 'v1')

      get "/api/v1/games/#{game.id}"

      expect(response).to have_http_status(:ok)
      expect(json_response[:game][:predictions][:gdo][:home_score]).to eq(75.5)
      expect(json_response[:game][:predictions][:gdo][:away_score]).to eq(70.2)
      expect(json_response[:game][:predictions][:gdo][:predicted_winner]).to eq('DUKE')
    end
  end
end