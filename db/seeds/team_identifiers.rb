
leagues = League.where(code: %w[mlb nfl ncaaf ncaam]).index_by(&:code)

teams_by_league = leagues.transform_values do |league|
  Team.where(league: league).index_by(&:code)
end

sources = DataSource.where(code: %w[sportsline mlb_api roto_wire]).index_by(&:code)


shared_sportsline_overrides = {
  "AAMU" => "ALAM",
  "ACU" => "ABIL",
  "AFA" => "AF",
  "AKR" => "AKRON",
  "ALBY" => "ALBANY",
  "APP" => "APLST",
  "ARST" => "ARKST",
  "ASU" => "ARIZST",
  "AUB" => "AUBURN",
  "BALL" => "BALLST",
  "BAY" => "BAYLOR",
  "BGSU" => "BGREEN",
  "BRY" => "BRYANT",
  "BSU" => "BOISE",
  "CCAR" => "CSTCAR",
  "CHAR" => "CHARLO",
  "CHAT" => "TNCHAT",
  "CHSO" => "CHARSO",
  "CIN" => "CINCY",
  "CMU" => "CMICH",
  "CONN" => "UCONN",
  "COOK" => "BTHU",
  "CP" => "CPOLY",
  "CSU" => "COLOST",
  "DEL" => "DE",
  "DSU" => "DELST",
  "EIU" => "EILL",
  "EMU" => "EMICH",
  "ETAM" => "TAMC",
  "EWU" => "EWASH",
  "FOR" => "FORD",
  "FUR" => "FURMAN",
  "FRES" => "FRESNO",
  "GASO" => "GAS",
  "GT" => "GATECH",
  "HAW" => "HAWAII",
  "HC" => "HOLY",
  "IDHO" => "IDAHO",
  "INST" => "INDST",
  "ISU" => "IOWAST",
  "JKST" => "JACKST",
  "JMU" => "JMAD",
  "JVST" => "JAXST",
  "KENT" => "KENTST",
  "KSU" => "KSTATE",
  "KSW" => "KENSAW",
  "KU" => "KANSAS",
  "LAM" => "LAMAR",
  "LOU" => "LVILLE",
  "LT" => "LATECH",
  "Linden" => "LINDEN",
  "M-OH" => "MIAOH",
  "MACK" => "MERMAK",
  "MCNS" => "MCNSE",
  "MER" => "MERCER",
  "MIA" => "MIAMI",
  "MIZZ" => "MIZZOU",
  "MNE" => "MAINE",
  "MORG" => "MORGAN",
  "MSST" => "MISSST",
  "MSU" => "MICHST",
  "MTST" => "MONST",
  "NA" => "NAL",
  "NCCU" => "NCCEN",
  "NEV" => "NEVADA",
  "NICH" => "NICHST",
  "NIU" => "NILL",
  "NMSU" => "NMEXST",
  "NW" => "NWEST",
  "OKST" => "OKLAST",
  "ORE" => "OREG",
  "ORST" => "OREGST",
  "OSU" => "OHIOST",
  "PEAY" => "AP",
  "PRST" => "PORTST",
  "PUR" => "PURDUE",
  "RMU" => "ROB",
  "RUTG" => "RUT",
  "SAC" => "SACST",
  "SCAR" => "SC",
  "SDKS" => "SDAKST",
  "SDSU" => "SDGST",
  "SELA" => "SELOU",
  "SEMO" => "SEMOST",
  "SFU" => "STRFPA",
  "SHSU" => "SAMST",
  "SIU" => "SIL",
  "SJSU" => "SJST",
  "STAN" => "STNFRD",
  "STON" => "STONYBRK",
  "SYR" => "CUSE",
  "TAMU" => "TXAM",
  "TAR" => "TRLST",
  "TEM" => "TEMPLE",
  "TEX" => "TEXAS",
  "TLSA" => "TULSA",
  "TNTC" => "TNTECH",
  "TOL" => "TOLEDO",
  "TTU" => "TXTECH",
  "TULN" => "TULANE",
  "TXSO" => "TEXSO",
  "TXST" => "TXSTSM",
  "UCA" => "CAR",
  "UCD" => "DAVIS",
  "ULM" => "LAMON",
  "UMD" => "MD",
  "UNCO" => "NCOLO",
  "UND" => "NDAK",
  "UNI" => "NIOWA",
  "UNM" => "NMEX",
  "UNT" => "NTEXAS",
  "URI" => "RI",
  "USA" => "SALA",
  "USF" => "SFLA",
  "USU" => "UTAHST",
  "UTM" => "TNMART",
  "UTSA" => "TXSA",
  "VAN" => "VANDY",
  "VT" => "VATECH",
  "WAG" => "WAGNER",
  "WCU" => "WCAR",
  "WEB" => "WBRST",
  "WEBB" => "GWEBB",
  "WIS" => "WISC",
  "WIU" => "WIL",
  "WKU" => "WKY",
  "WMU" => "WMICH",
  "WOF" => "WOFF",
  "WSU" => "WASHST"
}


overrides_by_league_and_source = {
  "mlb" => {
    "sportsline" => {
      "KCR" => "KC",
      "SDP" => "SD",
      "SFG" => "SF",
      "TBR" => "TB",
      "WSN" => "WAS"
    },
    "mlb_api" => {
      "ARI" => "AZ",
      "CHW" => "CWS",
      "KCR" => "KC",
      "SDP" => "SD",
      "SFG" => "SF",
      "TBR" => "TB",
      "WSN" => "WSH"
    },
    "roto_wire" => {
      "CHW" => "CWS",
      "KCR" => "KC",
      "SDP" => "SD",
      "SFG" => "SF",
      "TBR" => "TB",
      "WSN" => "WSH"
    }
    # fan_graphs has no overrides — all team codes match
  },
  "nfl" => {
    "sportsline" => {
      "GNB" => "GB",
      "JAX" => "JAC",
      "KAN" => "KC",
      "NOR" => "NO",
      "NWE" => "NE",
      "SFO" => "SF",
      "TAM" => "TB",
    }
  },
  "ncaaf" => {
    "sportsline" => shared_sportsline_overrides
  },
  "ncaam" => {
    "sportsline" => shared_sportsline_overrides
  }

}

# Step 4: Seed team identifiers
overrides_by_league_and_source.each do |league_code, source_overrides|
  league = leagues.fetch(league_code)
  teams  = teams_by_league.fetch(league_code)

  source_overrides.each do |source_code, team_overrides|
    source = sources.fetch(source_code)

    teams.each do |internal_code, team|
      external_code = team_overrides[internal_code] || internal_code

      TeamIdentifier.find_or_create_by!(
        team: team,
        league: league,
        data_source: source,
        external_code: external_code,
        active: true
      )
    end
  end
end