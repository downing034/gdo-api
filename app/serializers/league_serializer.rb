# frozen_string_literal: true

class LeagueSerializer < BaseSerializer
  def as_json
    {
      id: object.id,
      code: object.code,
      name: object.name,
      display_name: object.display_name,
      sport: object.sport.code
    }
  end
end