# frozen_string_literal: true

require 'rails_helper'

RSpec.describe 'Api::V1::Predictions', type: :request do
  describe 'POST /api/v1/predictions/simulate' do
    let(:sport) { create(:sport, :basketball) }
    let(:league) { create(:league, :ncaam, sport: sport) }
    let(:away_team) { create(:team, :duke) }
    let(:home_team) { create(:team, :unc) }

    before do
      league.teams << [away_team, home_team]
    end

    context 'with missing params' do
      it 'returns 400 when league is missing' do
        post '/api/v1/predictions/simulate', params: {
          away_team_code: away_team.code,
          home_team_code: home_team.code
        }

        expect(response).to have_http_status(:bad_request)
        expect(json_response[:error][:code]).to eq('bad_request')
      end

      it 'returns 400 for unsupported league' do
        post '/api/v1/predictions/simulate', params: {
          league: 'nhl',
          away_team_code: away_team.code,
          home_team_code: home_team.code
        }

        expect(response).to have_http_status(:bad_request)
        expect(json_response[:error][:code]).to eq('bad_request')
      end
    end

    context 'with invalid teams' do
      it 'returns 404 for unknown away team' do
        post '/api/v1/predictions/simulate', params: {
          league: 'ncaam',
          away_team_code: 'UNKNOWN',
          home_team_code: home_team.code
        }

        expect(response).to have_http_status(:not_found)
        expect(json_response[:error][:code]).to eq('not_found')
      end

      it 'returns 404 for unknown home team' do
        post '/api/v1/predictions/simulate', params: {
          league: 'ncaam',
          away_team_code: away_team.code,
          home_team_code: 'UNKNOWN'
        }

        expect(response).to have_http_status(:not_found)
        expect(json_response[:error][:code]).to eq('not_found')
      end
    end

    context 'with valid params', :slow do
      it 'returns prediction result' do
        post '/api/v1/predictions/simulate', params: {
          league: 'ncaam',
          away_team_code: away_team.code,
          home_team_code: home_team.code
        }

        expect(response).to have_http_status(:ok)
        expect(json_response[:prediction][:away_team][:code]).to eq('DUKE')
        expect(json_response[:prediction][:home_team][:code]).to eq('UNC')
        expect(json_response[:prediction][:away_team][:predicted_score]).to be_a(Numeric)
        expect(json_response[:prediction][:home_team][:predicted_score]).to be_a(Numeric)
        expect(json_response[:prediction][:spread]).to be_a(Numeric)
        expect(json_response[:prediction][:favorite]).to be_present
        expect(json_response[:prediction][:total]).to be_a(Numeric)
      end

      it 'handles neutral court' do
        post '/api/v1/predictions/simulate', params: {
          league: 'ncaam',
          away_team_code: away_team.code,
          home_team_code: home_team.code,
          neutral: true
        }

        expect(response).to have_http_status(:ok)
        expect(json_response[:prediction]).to be_present
      end
    end
  end
end