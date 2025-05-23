import lazyllm
from lazyllm.tools import SqlManager, SqlCall

table_info = {
    "tables": [{
        "name": "papers",
        "comment": "论文数据",
        "columns": [
            {
                "name": "id",
                "data_type": "Integer",
                "comment": "序号",
                "is_primary_key": True,
            },
            {"name": "title", "data_type": "String", "comment": "标题"},
            {"name": "author", "data_type": "String", "comment": "作者"},
            {"name": "subject", "data_type": "String", "comment": "领域"},
        ],
    }]
}

sql_manager = SqlManager("sqlite", None, None, None, None, db_name="papers.db", 
	tables_info_dict=table_info)
sql_llm = lazyllm.OnlineChatModule()
sql_call = SqlCall(sql_llm, sql_manager, 
	use_llm_for_sql_result=False)
query = "库中一共多少篇文章"
print(sql_call(query))
# >>> [{"total_papers": 100}]
