# frozen_string_literal: true

FactoryBot.define do
  factory :sport do
    sequence(:code) { |n| "sport_#{n}" }
    sequence(:name) { |n| "Sport #{n}" }

    trait :baseball do
      code { 'baseball' }
      name { 'Baseball' }
    end

    trait :basketball do
      code { 'basketball' }
      name { 'Basketball' }
    end

    trait :football do
      code { 'football' }
      name { 'Football' }
    end

    trait :hockey do
      code { 'hockey' }
      name { 'Hockey' }
    end
  end
end