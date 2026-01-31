# frozen_string_literal: true

RSpec.shared_examples 'not found response' do |path|
  it 'returns 404 for unknown resource' do
    get path

    expect(response).to have_http_status(:not_found)
    expect(json_response[:error][:code]).to eq('not_found')
  end
end

RSpec.shared_examples 'bad request response' do |method, path, params = {}|
  it 'returns 400 for bad request' do
    send(method, path, params: params)

    expect(response).to have_http_status(:bad_request)
    expect(json_response[:error][:code]).to eq('bad_request')
  end
end