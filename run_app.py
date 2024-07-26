from strep.elex.app import Visualization
from strep.index_and_rate import find_relevant_metrics

from run_paper_evaluation import database, rated_database, meta, boundaries, real_boundaries, references

# else interactive
database, metrics, xaxis_default, yaxis_default = find_relevant_metrics(database, meta)
db = {'DB': (rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references)}
app = Visualization(db)
server = app.server
app.run_server(debug=False, host='0.0.0.0', port=10000)
