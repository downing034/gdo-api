# frozen_string_literal: true

require 'rails_helper'

RSpec.describe Ncaam::SimulateService do
  let(:sport) { create(:sport, :basketball) }
  let(:league) { create(:league, :ncaam, sport: sport) }
  let(:away_team) { create(:team, :duke) }
  let(:home_team) { create(:team, :unc) }

  before do
    league.teams << [away_team, home_team]
  end

  describe '#call' do
    context 'with invalid teams' do
      it 'raises RecordNotFound for unknown away team' do
        service = described_class.new(
          away_team_code: 'UNKNOWN',
          home_team_code: home_team.code
        )

        expect { service.call }.to raise_error(ActiveRecord::RecordNotFound)
      end

      it 'raises RecordNotFound for unknown home team' do
        service = described_class.new(
          away_team_code: away_team.code,
          home_team_code: 'UNKNOWN'
        )

        expect { service.call }.to raise_error(ActiveRecord::RecordNotFound)
      end

      it 'raises RecordNotFound for team not in ncaam' do
        other_team = create(:team)

        service = described_class.new(
          away_team_code: other_team.code,
          home_team_code: home_team.code
        )

        expect { service.call }.to raise_error(ActiveRecord::RecordNotFound)
      end
    end

    context 'with valid teams', :slow do
      it 'returns prediction hash' do
        service = described_class.new(
          away_team_code: away_team.code,
          home_team_code: home_team.code
        )

        result = service.call

        expect(result[:away_team][:code]).to eq('DUKE')
        expect(result[:away_team][:location_name]).to eq('Duke')
        expect(result[:away_team][:nickname]).to eq('Blue Devils')
        expect(result[:away_team][:predicted_score]).to be_a(Numeric)

        expect(result[:home_team][:code]).to eq('UNC')
        expect(result[:home_team][:location_name]).to eq('North Carolina')
        expect(result[:home_team][:nickname]).to eq('Tar Heels')
        expect(result[:home_team][:predicted_score]).to be_a(Numeric)

        expect(result[:spread]).to be_a(Numeric)
        expect(result[:favorite]).to eq('DUKE').or eq('UNC')
        expect(result[:total]).to be_a(Numeric)
      end

      it 'handles lowercase team codes' do
        service = described_class.new(
          away_team_code: 'duke',
          home_team_code: 'unc'
        )

        result = service.call

        expect(result[:away_team][:code]).to eq('DUKE')
      end
    end
  end
end