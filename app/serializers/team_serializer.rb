# frozen_string_literal: true

class TeamSerializer < BaseSerializer
  def as_json
    {
      id: object.id,
      code: object.code,
      location_name: object.location_name,
      nickname: object.nickname,
      full_name: "#{object.location_name} #{object.nickname}",
      active: object.active
    }
  end
end