namespace :games do
  desc "Export stale games to CSV"
  task :export_stale, [:league_code] => :environment do |t, args|
    unless args[:league_code]
      puts "Error: league_code required"
      exit 1
    end
    
    ExportStaleGamesJob.perform_now(args[:league_code])
  end
end