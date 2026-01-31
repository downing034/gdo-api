Rails.application.routes.draw do
  namespace :api do
    namespace :v1 do
      resources :leagues, only: [:index, :show], param: :code do
        resources :teams, only: [:index], module: :leagues
      end
      resources :teams, only: [:show], param: :code
      resources :games, only: [:index, :show]
      
      resources :predictions, only: [] do
        collection do
          post :simulate
        end
      end

      get "daily_schedule", to: "daily_schedule#show"
    end
  end

  get "up" => "rails/health#show", as: :rails_health_check
end