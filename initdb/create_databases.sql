CREATE DATABASE demand_forecast;
CREATE DATABASE superset;
CREATE USER demand_forecast WITH ENCRYPTED PASSWORD 'demand_forecast';
GRANT ALL PRIVILEGES ON DATABASE demand_forecast TO demand_forecast;
GRANT ALL PRIVILEGES ON DATABASE superset TO demand_forecast;

\c superset
GRANT USAGE ON SCHEMA public TO demand_forecast;
GRANT CREATE ON SCHEMA public TO demand_forecast;

\c demand_forecast
GRANT USAGE ON SCHEMA public TO demand_forecast;
GRANT CREATE ON SCHEMA public TO demand_forecast;