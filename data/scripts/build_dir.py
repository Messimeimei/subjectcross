import os
from pathlib import Path
import re


def create_meta_data_directories():
    """
    在data/01meta_data目录下创建学科代码和完整名称对应的子目录
    如果目录已存在则跳过
    """
    # 定义学科代码和名称列表
    disciplines = [
        "0101 Philosophy",
        "0201 Theoretical Economics",
        "0202 Applied Economics",
        "0301 Law",
        "0302 Political Science",
        "0303 Sociology",
        "0304 Ethnology",
        "0401 Education",
        "0402 Psychology",
        "0403 Physical Education and Sport Science",
        "0502 Foreign Language and Literature",
        "0503 Journalism and Communication",
        "0601 Archaeology",
        "0602 History of China",
        "0603 World History",
        "0701 Mathematics",
        "0702 Physics",
        "0703 Chemistry",
        "0704 Astronomy",
        "0705 Geography",
        "0706 Atmospheric Science",
        "0707 Marine Science",
        "0708 Geophysics",
        "0709 Geology",
        "0710 Biology",
        "0711 Systems Science",
        "0712 History of Science and Technology",
        "0713 Ecology",
        "0714 Statistics",
        "0801 Mechanics",
        "0802 Mechanical Engineering",
        "0803 Optical Engineering",
        "0804 Instrumentation Science and Technology",
        "0805 Materials Science and Engineering",
        "0806 Metallurgical Engineering",
        "0807 Power Engineering and Engineering Thermophysics",
        "0808 Electrical Engineering",
        "0809 Electronic Science and Technology",
        "0810 Information and Communication Engineering",
        "0811 Control Science and Engineering",
        "0812 Computer Science and Technology",
        "0813 Architecture",
        "0814 Civil Engineering",
        "0815 Hydraulic Engineering",
        "0816 Surveying and Mapping",
        "0817 Chemical Engineering and Technology",
        "0818 Geological Resources and Geological Engineering",
        "0819 Mining Engineering",
        "0820 Oil and Natural Gas Engineering",
        "0821 Textile Science and Engineering",
        "0822 Light Industry Technology and Engineering",
        "0823 Transportation Engineering",
        "0824 Naval Architecture and Ocean Engineering",
        "0825 Aerospace Science and Technology",
        "0826 Armament Science and Technology",
        "0827 Nuclear Science and Technology",
        "0828 Agricultural Engineering",
        "0829 Forestry Engineering",
        "0830 Environmental Science and Engineering",
        "0831 Biomedical Engineering",
        "0832 Food Science and Engineering",
        "0833 Urban and Rural Planning",
        "0834 Landscape Architecture",
        "0835 Software Engineering",
        "0836 Biotechnology and Bioengineering",
        "0837 Safety Science and Engineering",
        "0839 Cyberspace Security",
        "0901 Crop Science",
        "0902 Horticulture",
        "0903 Agricultural Resource and Environment Sciences",
        "0904 Plant Protection",
        "0905 Animal Science",
        "0906 Veterinary Medicine",
        "0907 Forestry",
        "0908 Fisheries",
        "0909 Grassland Science",
        "1001 Basic Medicine",
        "1002 Clinical Medicine",
        "1003 Stomatology",
        "1004 Public Health and Preventive Medicine",
        "1005 Chinese Medicine",
        "1007 Pharmaceutical Science",
        "1008 Chinese Materia Medica",
        "1009 Special Medicine",
        "1010 Medical Technology",
        "1011 Nursing",
        "1201 Management Science and Engineering",
        "1202 Business Administration",
        "1203 Economics and Management of Agriculture and Forestry",
        "1204 Public Administration",
        "1205 Library and Information Science & Archive Management",
        "1301 Art Theory",
        "1302 Music and Dance",
        "1303 Drama Film and Television",
        "1304 Fine Art",
        "1305 Design",
        "1400 Cross-field"
    ]

    # 获取当前脚本所在目录的父目录
    current_dir = Path(__file__).parent
    base_dir = current_dir.parent  # 项目根目录

    # 构建目标目录路径
    meta_data_dir = base_dir / "data" / "01meta_data"

    def sanitize_filename(name):
        """清理文件名，移除非法字符"""
        # 替换Windows文件名中不允许的字符
        invalid_chars = r'[<>:"/\\|?*]'
        name = re.sub(invalid_chars, '_', name)
        # 移除开头和结尾的空格和点号
        name = name.strip('. ')
        # 限制文件名长度（Windows最大255字符，这里设为100以保持可读性）
        if len(name) > 100:
            name = name[:100]
        return name

    try:
        # 创建主目录（如果不存在）
        meta_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"创建主目录: {meta_data_dir}")

        # 为每个学科创建子目录
        created_count = 0
        existing_count = 0

        for discipline in disciplines:
            # 使用完整的学科代码和名称作为目录名
            directory_name = sanitize_filename(discipline)
            discipline_dir = meta_data_dir / directory_name

            # 创建学科目录
            if not discipline_dir.exists():
                discipline_dir.mkdir(parents=False, exist_ok=True)
                print(f"创建学科目录: {discipline_dir}")
                created_count += 1
            else:
                print(f"目录已存在: {discipline_dir}")
                existing_count += 1

        print(f"\n目录创建完成!")
        print(f"新创建目录: {created_count} 个")
        print(f"已存在目录: {existing_count} 个")
        print(f"总计目录: {len(disciplines)} 个")

        # 显示前几个创建的目录作为示例
        print(f"\n目录示例:")
        for i, discipline in enumerate(disciplines[:5]):
            directory_name = sanitize_filename(discipline)
            print(f"  {i + 1}. {directory_name}")
        if len(disciplines) > 5:
            print(f"  ... 和其他 {len(disciplines) - 5} 个目录")

    except Exception as e:
        print(f"创建目录时出错: {e}")


if __name__ == "__main__":
    create_meta_data_directories()