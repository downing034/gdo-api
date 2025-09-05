DataSource.create!([
  # mlb, ncaaf, ncaam, nfl
  {
    code: "gdo",
    name: "Gameday Oracle",
    base_url: ""
  },
  # mlb, ncaaf, ncaam, nfl
  {
    code: "sportsline",
    name: "Sportsline",
    base_url: "https://www.sportsline.com/"
  },
  # mlb  
  {
    code: "mlb_api",
    name: "MLB StatsAPI",
    base_url: "https://statsapi.mlb.com"
  },
  # mlb
  {
    code: "fan_graphs",
    name: "Fan Graphs",
    base_url: "https://www.fangraphs.com"
  },
  # mlb
  {
    code: "roto_grinders",
    name: "Roto Grinders",
    base_url: "https://rotogrinders.com"
  },
  # mlb
  {
    code: "roto_wire",
    name: "Roto Wire",
    base_url: "https://www.rotowire.com"
  },
  {
    code: "espn",
    name: "ESPN Sports API",
    base_url: "https://site.api.espn.com/apis/site/v2"
  },
  # ncaaf
  {
    code: "sports_reference",
    name: "Sport Reference",
    base_url: "https://www.sports-reference.com/cfb"
  },
  # nfl
  {
    code: "pro_football_reference",
    name: "Pro Football Reference",
    base_url: "https://www.pro-football-reference.com"
  },
  # ncaam
  {
    code: "ken_Pom",
    name: "Ken Pom",
    base_url: "https://kenpom.com"
  },
  # ncaam
  {
    code: "barttorvik",
    name: "Bart Torvik",
    base_url: "https://barttorvik.com/trankpre.php"
  }
])