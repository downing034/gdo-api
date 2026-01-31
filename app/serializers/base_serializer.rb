# frozen_string_literal: true

class BaseSerializer
  def initialize(object, options = {})
    @object = object
    @options = options
  end

  def as_json
    raise NotImplementedError
  end

  def to_json
    as_json.to_json
  end

  private

  attr_reader :object, :options

  def serialize_collection(collection, serializer_class)
    collection.map { |item| serializer_class.new(item, options).as_json }
  end
end