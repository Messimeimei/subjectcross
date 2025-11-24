# wos数据使用教程v1

版本：v1

日期：2025年9月29日

author: fkliu

---

# 数据集使用需知

1.  存储：
    
    1.  （完整）存储在Starrocks数据库，用于大批量计算，批量查询
        
        1.  表crossref.meta\_data\_core
            
    2.  （完整）存储在ES数据库中，用于小批量多次高速查询（同步Starrocks）
        
        1.  索引名称为meta\_dat\_core
            
    3.  （非常复杂详细的数据集）18年及之前的原始数据存储在81服务器psql中
        
2.  数据统计看板
    
    1.  superset平台 用于查看数据统计图表，自定义图表和有限制的SQL查询
        
        1.  连接上述两个存储数据库
            
    2.  datahub平台
        
        1.  元数据平台，描述数据库字段
            
        2.  但目前看常用功能被superset覆盖
            
3.  使用方式分类
    
    1.  大量数据计算、导出；大批量数据查询
        
        1.  截止到教程写作时间，直接在Starrocks数据库中计算
            
    2.  高频快速响应查询：
        
        1.  ES API，详情见后文
            
    3.  文档级别检索
        
        1.  ES API
            

# ES使用数据

### ES API教程

ES 条件查询 [【ES】ElasticSearch 数据库之查询操作 从入门＞实践＞精通 一篇文章包含ES的所有查询语法\_怎么查看es数据-CSDN博客](https://blog.csdn.net/huangtenglong/article/details/151933542)

ES 批量导出 [es滚动查询分析和使用步骤示例详解\_相关技巧\_脚本之家](https://www.jb51.net/program/298488kwm.htm)

### WOS ES API示例

```sh
import requests
import json

# Elasticsearch 服务地址（注意使用 9200 端口）
es_url = "http://172.17.60.121:9200/meta\_data\_core/\_search"

# 查询体,查询体可以直接百度，有多种检索策略
query_body = {
    "query": {
        "bool": {
            "must": [
                {"match": {"pub_year": "subsurface"}}
            ]
        }
    }
}

try:
    # 发送 GET 请求
    response = requests.get(
        es_url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(query_body),
        # auth=('elastic', 'your_password')  # 有认证时取消注释并填写账号密码
    )
    
    # 检查响应状态
    response.raise_for_status()
    
    # 解析并打印结果
    result = response.json()
    print("查询成功，结果如下：")
    print(json.dumps(result, indent=2, ensure_ascii=False))

except requests.exceptions.RequestException as e:
    print(f"请求失败：{e}")

```

# Starrocks使用数据

*   python mysqlclient连接数据库使用
    
*   服务器进入数据库直接进行sql操作
    
    *   服务器ip：172.17.60.122 
        
    *   账号名：user\_crossref\_read （该账号不具备修改权限）
        
    *   密码：JT7JHMPq@Uux9MYN
        

# 常见需要注意的问题

#### 1.对被引次数要求高不高，必须实时和wos检索到的完全一致吗？

下面是测试本数据集于wos上被引次数准确率的统计数据，除非要求完全一致，否则均可作为wos实时被引次数使用

实验室数据库准确率测试：

![Pasted image 20250928184321.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarbA62mB8qXx/img/d53931b5-a6be-4873-8682-016e2463c356.png)

![Pasted image 20250928184345.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarbA62mB8qXx/img/0908e8f1-eed3-4d04-a6e4-3b2a338d2928.png)

# WOS字段说明

*   ES和Starrocks中meta\_data\_core为同步数据，单字段暂未完全同步，ES中略有缺失 且均为预处理后的字段
    
*   以下是按照一般逻辑顺序整理后的内容：
    

| 大分类 | StarRocks 字段名 | StarRocks 类型 | ES 对应字段名 | ES 字段类型 | ES 中是否存在 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| **1. 文献标识相关** | wos\_id | varchar | wos\_id | keyword | 是 | 文献在 WoS 平台的唯一标识，双方字段名、含义完全一致，ES 用于精确匹配 / 聚合 |
|  | doi | varchar | doi | keyword | 是 | 文献 DOI 编号（数字对象标识符），双方字段名、含义完全一致，ES 用于精确匹配 |
| **2. 文献基本信息** | title | varchar | title |  | 是 | 文献标题，双方含义一致，ES 中`text`类型支持标题全文检索 |
|  | abstract | varchar | abstract |  | 是 | 文献摘要，双方含义一致，ES 中`text`类型支持摘要全文检索 |
|  | language | varchar | language |  | 是 | 文献语言（如英文、中文），双方含义一致，ES 用于语言维度精确聚合 |
|  | publication\_type | varchar | publication\_type |  | 是 | 对应 WoS 导出字段 PT（文献出版类型），双方含义一致，ES 用于精确匹配 / 聚合 |
|  | doc\_type | varchar | doc\_type |  | 是 | 对应 WoS 导出字段 DT（文献文档类型），双方含义一致，ES 用于精确匹配 / 聚合 |
|  | wos\_category | varchar | wos\_category |  | 是 | WoS 学科分类（如计算机科学、医学），双方含义一致，ES 支持分类检索 |
|  | research area | varchar | research\_area |  | 是 | 文献研究领域，StarRocks 字段含空格，ES 字段用下划线（research\_area），含义完全对应，ES 支持领域检索 |
|  | wos\_index | varchar | wos\_index |  | 是 | WoS 索引类型（如 SCI-E、SSCI、AHCI），双方含义一致，ES 支持索引类型检索 |
| **3. 发表信息** | journal\_full\_name | varchar | journal\_full\_name |  | 是 | 文献所属期刊全名，双方含义一致，ES 中`text`类型支持期刊名全文检索 |
|  | issn | varchar | issn |  | 是 | 期刊印刷版 ISSN 号，双方含义一致，ES 用于期刊精确匹配 |
|  | eissn | varchar | eissn |  | 是 | 期刊电子版 ISSN 号，双方含义一致，ES 用于电子期刊精确匹配 |
|  | pub\_year | varchar | pub\_year |  | 是 | 文献发表年份，双方含义一致，ES 用于年份维度精确聚合（如按年统计发文量） |
|  | pub\_date | varchar | pub\_date |  | 是 | 文献发表月 / 日（如 05-12），双方含义一致，ES 中`index:false`仅存储不参与检索 |
|  | early\_access | varchar | early\_access |  | 是 | 文献提前在网络公开的时间，双方含义一致，ES 中`index:false`仅存储不参与检索 |
| **4. 作者及单位信息** | author\_full\_name | varchar | author\_full\_name |  | 是 | 作者全名（如 Zhang San），双方含义一致，ES 中`text`类型支持作者名检索 |
|  | author\_abbr\_name | varchar | author\_abbr\_name |  | 是 | 作者缩写名（如 Zhang S），双方含义一致，ES 中`text`类型支持缩写名检索 |
|  | author\_address | varchar | author\_address |  | 是 | 作者与地址对应字典（如 “Zhang S: Peking Univ, Beijing”），双方含义一致，ES 不参与检索 |
|  | affiliation | varchar | affiliation |  | 是 | 作者所属单位（如北京大学），双方含义一致，ES 支持单位名检索 |
|  | corresponding\_author | varchar | corresponding\_author |  | 是 | 通讯作者（负责接收期刊反馈的作者），双方含义一致，ES 支持通讯作者检索 |
|  | country | varchar | country |  | 是 | 作者所属国家（如 China、USA），双方含义一致，ES 用于国家维度精确聚合 |
| **5. 补充 / 其他信息** | reference | varchar | reference |  | 是 | 文献参考文献列表，双方含义一致，ES 用于参考文献精确匹配 |
|  | author\_keywords | varchar | author\_keywords |  | 是 | ES 中暂不存在该字段，仅 StarRocks 表包含（文献作者自主标注的关键词） |
|  | keyword\_plus | varchar | keyword\_plus |  | 是 | ES 中暂不存在该字段，仅 StarRocks 表包含（WoS 平台自动补充的扩展关键词） |
|  | wos\_c3 | varchar | wos\_c3 |  | 是 | ES 中暂不存在该字段，仅 StarRocks 表包含（WoS 核心合集分类标识，作者单位原始字段数据C3） |