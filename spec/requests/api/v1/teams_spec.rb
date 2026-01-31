# frozen_string_literal: true

require 'rails_helper'

RSpec.describe 'Api::V1::Teams', type: :request do
  describe 'GET /api/v1/teams/:code' do
    it_behaves_like 'not found response', '/api/v1/teams/UNKNOWN'

    it 'returns a team by code' do
      team = create(:team, :duke)

      get '/api/v1/teams/DUKE'

      expect(response).to have_http_status(:ok)
      expect(json_response[:team][:code]).to eq('DUKE')
      expect(json_response[:team][:full_name]).to eq('Duke Blue Devils')
    end

    it 'handles lowercase code' do
      team = create(:team, :duke)

      get '/api/v1/teams/duke'

      expect(response).to have_http_status(:ok)
      expect(json_response[:team][:code]).to eq('DUKE')
    end
  end

  describe 'GET /api/v1/leagues/:league_code/teams' do
    it_behaves_like 'not found response', '/api/v1/leagues/unknown/teams'

    it 'returns teams for a league' do
      sport = create(:sport, :basketball)
      league = create(:league, :ncaam, sport: sport)
      duke = create(:team, :duke)
      unc = create(:team, :unc)
      league.teams << [duke, unc]

      get "/api/v1/leagues/#{league.code}/teams"

      expect(response).to have_http_status(:ok)
      expect(json_response[:league]).to eq('ncaam')
      expect(json_response[:teams].length).to eq(2)
    end

    it 'excludes inactive teams' do
      sport = create(:sport, :basketball)
      league = create(:league, :ncaam, sport: sport)
      duke = create(:team, :duke)
      inactive = create(:team, :inactive)
      league.teams << [duke, inactive]

      get "/api/v1/leagues/#{league.code}/teams"

      expect(response).to have_http_status(:ok)
      expect(json_response[:teams].length).to eq(1)
    end
  end
end