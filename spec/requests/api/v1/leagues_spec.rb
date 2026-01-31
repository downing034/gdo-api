# frozen_string_literal: true

require 'rails_helper'

RSpec.describe 'Api::V1::Leagues', type: :request do
  describe 'GET /api/v1/leagues' do
    it 'returns empty array when no leagues exist' do
      get '/api/v1/leagues'

      expect(response).to have_http_status(:ok)
      expect(json_response[:leagues]).to eq([])
    end

    it 'returns active leagues' do
      sport = create(:sport, :basketball)
      league = create(:league, :ncaam, sport: sport)
      create(:league, :inactive, sport: sport)

      get '/api/v1/leagues'

      expect(response).to have_http_status(:ok)
      expect(json_response[:leagues].length).to eq(1)
      expect(json_response[:leagues].first[:code]).to eq('ncaam')
    end
  end

  describe 'GET /api/v1/leagues/:code' do
    it_behaves_like 'not found response', '/api/v1/leagues/unknown'

    it 'returns a league by code' do
      sport = create(:sport, :basketball)
      league = create(:league, :ncaam, sport: sport)

      get "/api/v1/leagues/#{league.code}"

      expect(response).to have_http_status(:ok)
      expect(json_response[:league][:code]).to eq('ncaam')
      expect(json_response[:league][:name]).to eq("NCAA Men's Basketball")
    end
  end
end