import pandas as pd
import numpy as np
import math
import datetime
import dateutil.relativedelta as daterel
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import tushare as ts
import sys

# 潜在问题：merge之后，index是否会重复？
# make the figures display Chinese, require "import matplotlib"
matplotlib.rc("font", family='DengXian')

# 设置token, 只需要在第一次或者token失效后调用
# ts.set_token('22d922459ae67ae13d743574825591daf643eb2df520d2991f921232')
pro = ts.pro_api()  # 初始化pro接口


def has_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


'''
def query_df_v1(df, col_name, target_value):
    """query the index of the target_value in col_name,
    and return it if there exists exactly one.\n
    Note: the returned data is the first element of idx"""
    idx = df[df[col_name] == target_value].index.tolist()
    check_num_idx(df, col_name, idx)
    return idx[0]
'''


def query_df(data_frame, query_col, target_value, output_col):
    """query the data_frame, find the target_value in col_name,
    and return it if there exists exactly one.\n
    Note: the returned data is the first element of idx"""
    # get the index of the target_value
    idx = data_frame[data_frame[query_col] == target_value].index.tolist()
    check_num_idx(data_frame, query_col, idx)
    # split output_col into a list: use comma as the separator, and remove the blank
    output_list = [s.strip() for s in output_col.split(',')]
    if len(output_list) == 1:
        return data_frame.loc[idx[0], output_list[0]]
    else:
        result_list = []
        for i in range(len(output_list)):
            result_list.append(data_frame.loc[idx[0], output_list[i]])
        return result_list


def check_num_idx(check_df, check_col_name, check_idx):
    # pd.set_option('display.max_columns', None)
    if len(check_idx) < 1:
        print('The {} not found!'.format(check_col_name))
        sys.exit()
    elif len(check_idx) > 1:
        print('Multiple (>=2) {} found!'.format(check_col_name))
        print(check_df.loc[check_idx, :])  # print the multiple rows
        sys.exit()
    # else:
    #     print('\n Queried row is: ')
    #     print(check_df.loc[check_idx, :])


class Asset(object):

    def __init__(self, asset_info, date_interval=('', '')):
        self.code, self.name = self.get_code_name(asset_info)
        self.date_interval = date_interval
        self.data = self.get_data()
        self.check_data()

    def get_code_name(self, asset_info):
        return '', ''

    def get_data(self):
        pass

    def check_data(self):
        """Remove the reduplicative dates, and sort the dates. \n
        Why not integrated in get_data()? Check_data is a unified operation. """
        # remove reduplicative rows with the same trade_date
        self.data = self.data.drop_duplicates(subset='trade_date')
        self.data = self.data.sort_values(by='trade_date')  # sort by date

    def normalize_data(self):
        """Normalize the data to [0, 1], examining the correlation."""
        max_close = self.data['close'].max()  # don't use max(df['close']), it may return 'nan'
        min_close = self.data['close'].min()
        self.data['close'] = (self.data['close']-min_close)/(max_close-min_close)

    def reset_start_data(self):
        """Set the starting point of the data to 1,
        comparing the potential of different stocks."""
        self.data['close'] = self.data['close']/self.data.loc[self.data.index[0], 'close']

    def moving_average(self, ma_period):
        """For self.data, add the moving average of the close price to the last column.\n
        PARAMETER: ma_period (a list, e.g., [5, 10, 20]).\n
        The added columns: 'ma5', 'ma10', ..."""
        for j in range(len(self.data.index)):
            for k in ma_period:
                if j + 1 >= k:
                    self.data.loc[self.data.index[j], 'ma' + str(k)] = \
                        np.mean(self.data.loc[self.data.index[j-k+1]:self.data.index[j], 'close'])

    def show(self, plot_field=('close',)):
        for j in plot_field:
            plt.plot(self.data['trade_date'], self.data[j],
                     label=self.name + '_' + j, linewidth=1)

        # set the density of the xtick
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(self.data) / 20))

        plt.xticks(rotation=45)  # set the angle of texts in xtick
        # plt.legend(loc=0)
        plt.legend(bbox_to_anchor=(0.5, 1.03), loc='lower center', borderaxespad=0)
        plt.tight_layout()  # adjust figure margin
        plt.show()


class Stock(Asset):

    # noinspection PyMethodMayBeStatic
    def get_code_name(self, asset_info):  # convert Chinese name to ts_code
        # get the information of all the stocks
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
        if has_chinese(asset_info):
            name = asset_info
            code = query_df(data_frame=df, query_col='name',
                            target_value=name, output_col='ts_code')
        else:
            code = asset_info
            name = query_df(data_frame=df, query_col='ts_code',
                            target_value=code, output_col='name')
        return code, name

    def get_data(self):
        df = ts.pro_bar(
            ts_code=self.code,
            asset='E',
            adj='qfq',
            start_date=self.date_interval[0],
            end_date=self.date_interval[1])
        return df


class Index(Asset):

    # noinspection PyMethodMayBeStatic
    def get_code_name(self, asset_info):
        df = pro.index_basic(market='', fields='ts_code,name')  # list of all indexes
        if has_chinese(asset_info):
            name = asset_info
            code = query_df(data_frame=df, query_col='name',
                            target_value=name, output_col='ts_code')
        else:
            code = asset_info
            name = query_df(data_frame=df, query_col='ts_code',
                            target_value=code, output_col='name')
        return code, name

    def get_data(self):
        df = ts.pro_bar(
            ts_code=self.code,
            asset='I',
            start_date=self.date_interval[0],
            end_date=self.date_interval[1])
        return df


class ETF(Asset):

    # noinspection PyMethodMayBeStatic
    def get_code_name(self, asset_info):
        df = pro.fund_basic(market='E')
        if has_chinese(asset_info):
            name = asset_info
            code = query_df(data_frame=df, query_col='name',
                            target_value=name, output_col='ts_code')
        else:
            code = asset_info
            name = query_df(data_frame=df, query_col='ts_code',
                            target_value=code, output_col='name')
        return code, name

    def get_data(self):
        df = pro.fund_nav(
            ts_code=self.code,
            market='E',
            start_date=self.date_interval[0],
            end_date=self.date_interval[1])
        df.rename(columns={'nav_date': 'trade_date', 'unit_nav': 'close'}, inplace=True)
        return df


class Future(Asset):
    pass


class Option(Asset):
    """Get the data of an option (self.data) from the ts_code.
    Attributes:
        expiration_date: string, e.g., '20230125'
        opt_type: '购' or '沽'
        strike_price: string, e.g., '2.60'"""

    def __init__(self, asset_info, date_interval=('', '')):
        super().__init__(asset_info, date_interval)
        self.expiration_date, self.opt_type, self.strike_price = self.name_to_info()

    def get_code_name(self, asset_info):
        # query all the option information
        df = pd.DataFrame(columns=['ts_code', 'name'])
        for opt_exchg in {'CFFEX', 'SSE', 'SZSE', 'SHFE', 'CZCE', 'DCE'}:
            cur_df = pro.opt_basic(exchange=opt_exchg, fields='ts_code,name')  # list of all options
            df = pd.concat([df, cur_df])
            df.index = range(len(df))  # reset the index. or use the following method
            # df.reset_index(drop=True, inplace=True)

        # query the disired information
        if isinstance(asset_info, dict):  # generate from a dictionary
            # unify the name of opt_type
            if asset_info['opt_type'].lower() in {'c', 'call', '购', '认购'}:
                asset_info['opt_type'] = '购'
            elif asset_info['opt_type'].lower() in {'p', 'put', '沽', '认沽'}:
                asset_info['opt_type'] = '沽'
            # query the dataframe
            idx = df[(df['name'].str.contains(asset_info['asset_name'].upper()) |
                      df['name'].str.contains(asset_info['asset_name'].lower())) &
                     df['name'].str.contains(asset_info['opt_type']) &
                     df['name'].str.contains(asset_info['strike_price']) &
                     df['name'].str.contains(asset_info['expiration']) &
                     df['name'].str.contains(asset_info['opt_tips'])].index.tolist()
        elif has_chinese(asset_info):  # generate from the name
            idx = df[df['name'] == asset_info].index.tolist()
        else:  # generate from the code
            idx = df[df['ts_code'] == asset_info].index.tolist()

        # check and assign value
        check_num_idx(df, 'code_name', idx)  # if len(opt_idx)!=1, exit(),
        code = df.loc[idx[0], 'ts_code']  # !!! '[0]' is required
        name = df.loc[idx[0], 'name']
        return code, name

    def get_data(self):
        # query the data of the option
        df = pro.opt_daily(
            ts_code=self.code,
            start_date=self.date_interval[0],
            end_date=self.date_interval[1])
        return df

    def name_to_info(self):
        # get the expiration date
        idx = self.name.find('期权')
        # expiration_date = self.name[idx + 2:idx + 6]
        the_year = int('20'+self.name[idx + 2:idx + 4])
        the_month = int(self.name[idx + 4:idx + 6])
        first_day = datetime.date(the_year, the_month, 1)
        if 'ETF' in self.name.upper():
            # the fourth wednesday
            expiration_date = first_day + daterel.relativedelta(day=0, weekday=daterel.WE(4))
        else:
            # the third friday
            expiration_date = first_day + daterel.relativedelta(day=0, weekday=daterel.FR(3))
        expiration_date = expiration_date.strftime('%Y%m%d')  # convert to str

        # get the opt_type
        if '购' in self.name:
            opt_type = '购'
            idx = self.name.find('购')
        elif '沽' in self.name:
            opt_type = '沽'
            idx = self.name.find('沽')
        else:
            print('Opttype not found')
            sys.exit()

        # get the strike price
        strike_price = self.name[idx + 1:]
        return expiration_date, opt_type, strike_price


class OptionAdv(Option):
    """align date?
    Get the data of an option (self.data) from a dictionary which
    contains some key information (e.g., asset name, strike price...), and get the data of the
    corresponding asset (self.asset_data).\n
    Compute the intrinsic and time value of the option.
    """

    def __init__(self, asset_info, date_interval):  # noqa
        super().__init__(asset_info, date_interval)
        self.asset = self.get_asset()
        self.intri_time_value()
        self.compute_margin()

    def get_asset(self):
        """generate the codes of the asset and the corresponding option"""
        # get the asset name, which lies before '期权'
        idx = self.name.find('期权')
        asset_name = self.name[:idx]
        asset_name = asset_name.upper()  # make sure 'ETF' is upper case
        if '指数' in asset_name:
            asset_name = asset_name[:-2]  # remove '指数' at the end
            if asset_name == '沪深300':  # two 沪深300 (another is 399300.SZ), we need 000300.SH
                asset_info = '000300.SH'
            else:
                asset_info = asset_name
            return Index(asset_info, self.date_interval)
        elif 'ETF' in asset_name:
            if asset_name == '华泰柏瑞沪深300ETF':
                asset_info = '510300.SH'
            elif asset_name == '嘉实沪深300ETF':
                asset_info = '159919.SZ'
            elif '创业板ETF' in asset_name:
                asset_info = '159915.SZ'
            elif '华夏上证50ETF' in asset_name:
                asset_info = '510050.SH'
            else:
                asset_info = asset_name
            return ETF(asset_info, self.date_interval)

    def intri_time_value(self):
        """compute the intrinsic value and time value,
        and add to the last column for each dataframe in data_list"""
        # dummy column to compute the intri_val, such that the intri_val
        # doesn't exceed the trade dates of the options
        self.data['strike'] = float(self.strike_price)
        # align the dates
        [self.data, self.asset.data] = align_date([self.data, self.asset.data])
        # compute
        df = self.asset.data['close'] - self.data['strike']  # the result is a series
        if self.opt_type == '购':
            self.data['intri_val'] = (df.abs() + df) / 2
        elif self.opt_type == '沽':
            self.data['intri_val'] = (df.abs() - df) / 2
        self.data['time_val'] = self.data['close'] - self.data['intri_val']

    # noinspection PyMethodMayBeStatic
    def compute_margin(self):
        multiplier = 100
        adjust_factor = 0.15
        guarantee_factor = 0.5
        for j in range(len(self.data.index)):
            pre_settle = self.data.loc[self.data.index[j], 'pre_settle']
            # note: self.data and self.asset.data should be aligned by date
            asset_pre_close = self.asset.data.loc[self.asset.data.index[j], 'pre_close']
            if self.opt_type == '购':
                out_of_the_money = max((float(self.strike_price) - asset_pre_close)*multiplier, 0)
                temp = asset_pre_close*multiplier*adjust_factor
                margin = pre_settle*multiplier+max(temp-out_of_the_money, guarantee_factor*temp)
            elif self.opt_type == '沽':
                out_of_the_money = max(asset_pre_close - (float(self.strike_price))*multiplier, 0)
                temp = multiplier * adjust_factor
                margin = pre_settle*multiplier + max(asset_pre_close*temp-out_of_the_money,
                                                     guarantee_factor*(float(self.strike_price)*temp))
            else:
                print('self.opt_type is wrong!')
                sys.exit()
            self.data.loc[self.data.index[j], 'margin'] = margin
        for j in range(1, len(self.data.index)):
            self.data.loc[self.data.index[j], 'pre_margin'] = \
                self.data.loc[self.data.index[j-1], 'margin']

    def show(self, plot_field=('intri_val', 'time_val')):
        # plot the option
        # plot the close
        plt.plot(self.data['trade_date'], self.data['close'],
                 label=self.name + '_' + 'close',
                 marker='s', markersize=3, color='cadetblue',
                 linestyle='solid', linewidth=1, zorder=10)
        # plot the other fields
        for j in plot_field:
            if j == 'intri_val':
                plt.plot(self.data['trade_date'], self.data[j],
                         label=self.name + '_' + j,
                         marker='P', markersize=3,
                         linestyle='dashed', linewidth=1)
            elif j == 'time_val':
                plt.plot(self.data['trade_date'], self.data[j],
                         label=self.name + '_' + j,
                         marker='D', markersize=3, markerfacecolor='none',
                         linestyle='dashdot', linewidth=1)
                plt.legend(bbox_to_anchor=(0, 1.02), loc='lower left', borderaxespad=0)
            else:
                plt.plot(self.data['trade_date'], self.data[j],
                         label=self.name + '_' + j,
                         linestyle='solid', linewidth=1)

        plt.legend(bbox_to_anchor=(0, 1.02), loc='lower left', borderaxespad=0)
        # plt.gca().invert_yaxis()
        plt.xticks(rotation=45)  # set the angle of texts in xtick

        # plot the asset
        plt.twinx()
        plt.plot(self.asset.data['trade_date'], self.asset.data['close'],
                 label=self.asset.name + '_' + 'close',
                 marker='.', markersize=4,
                 color='k', linestyle='solid', linewidth=1)
        plt.legend(bbox_to_anchor=(1, 1.02), loc='lower right', borderaxespad=0)
        plt.gca().invert_yaxis()

        # plot the strike price
        guide_line = float(self.strike_price)
        plt.axhline(y=guide_line, color='k', linestyle='solid', linewidth=0.5)
        if self.opt_type == '购':
            guide_line *= 1.0007
        elif self.opt_type == '沽':
            guide_line *= 0.9993
        plt.axhline(y=guide_line, color='k', linestyle=(0, (5, 5)), linewidth=0.5)

        # set the density of the xtick
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(self.data) / 20))

        plt.tight_layout()
        plt.show()


class HSGT(Asset):

    def get_code(self):
        self.code = 'moneyflow_hsgt'

    def get_data(self):
        self.data = pro.query(
            'moneyflow_hsgt', start_date=self.date_interval[0], end_date=self.date_interval[1])
        self.data = self.data.sort_values(by='trade_date')  # sort by date

    def show(self, plot_field=('north_money',)):
        for j in plot_field:
            plt.bar(self.data['trade_date'], self.data[j],
                    label=self.name + '_' + j, color='deepskyblue')

        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(self.data) / 20))
        plt.xticks(rotation=45)  # set the angle of texts in xtick
        plt.legend(loc=0)
        plt.show()


class Combine(Asset):

    def __init__(self, obj_list, weight_list):  # noqa
        self.obj_list = obj_list
        self.weight_list = weight_list
        self.get_name()
        if isinstance(obj_list[0], Option):
            # for options, the dates need to aligned with the corresponding asset
            for i in range(len(obj_list)):
                [self.obj_list[i].data, self.obj_list[0].asset_data] = \
                    align_date([self.obj_list[i].data, self.obj_list[0].asset_data])
        else:
            align_obj_date(self.obj_list)
        pd.set_option('display.max_rows', None)
        self.get_data()

    def get_name(self):
        self.name = '组合_'
        for i in range(len(self.obj_list)):
            self.name = self.name + str(self.weight_list[i]) + '-'
        self.name = self.name[:-1]
        self.name += '_'
        for i in range(len(self.obj_list)):
            self.name = self.name + self.obj_list[i].name + '-'
        self.name = self.name[:-1]

    def get_data(self):
        df = pd.DataFrame(index=self.obj_list[0].data.index, columns=self.obj_list[0].data.columns)
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                if df.columns[j] in {'trade_date', 'exchange'}:
                    df.iloc[i, j] = self.obj_list[0].data.iloc[i, j]
                elif df.columns[j] in {'ts_code'}:
                    df.iloc[i, j] = 'combine'
                else:
                    temp = 0
                    for k in range(len(self.obj_list)):
                        temp += self.weight_list[k]*self.obj_list[k].data.iloc[i, j]
                    df.iloc[i, j] = temp
        self.data = df

    def show_opt(self, decomp_type='close'):
        """When combining options, plot the asset and the options.\n
        Paramater:
                decomp_type: 'close' or 'time' (default='close'), how to decompose the combined option data. """
        self.name = check_name_len(self.name, 28)  # break the long name into several lines

        # plot the combined options
        plt.plot(self.data['trade_date'], self.data['close'],
                 label=self.name + '_' + 'close',
                 marker='s', markersize=3, color='cadetblue',
                 linestyle='solid', linewidth=1, zorder=10)

        # plot the options
        if decomp_type == 'close':
            for i in range(len(self.obj_list)):
                if self.obj_list[i].opt_type == '购':
                    plt.plot(self.obj_list[i].data['trade_date'], self.obj_list[i].data['close'],
                             label=self.obj_list[i].name + '_' + 'close',
                             marker='v', markersize=3, markerfacecolor='none',
                             linestyle='dashed', linewidth=1)
                elif self.obj_list[i].opt_type == '沽':
                    plt.plot(self.obj_list[i].data['trade_date'], self.obj_list[i].data['close'],
                             label=self.obj_list[i].name + '_' + 'close',
                             marker='^', markersize=3, markerfacecolor='none',
                             linestyle='dashdot', linewidth=1)
        elif decomp_type == 'time':
            plt.plot(self.data['trade_date'], self.data['intri_val'],
                     label='组合_intri_val',
                     marker='P', markersize=3,
                     linestyle='dashed', linewidth=1)
            plt.plot(self.data['trade_date'], self.data['time_val'],
                     label='组合_time_val',
                     marker='D', markersize=3, markerfacecolor='none',
                     linestyle='dashdot', linewidth=1)

        plt.legend(bbox_to_anchor=(0, 1.02), loc='lower left', borderaxespad=0)
        # plt.gca().invert_yaxis()
        plt.xticks(rotation=45)  # set the angle of texts in xtick

        # plot the asset
        plt.twinx()
        plt.plot(self.obj_list[0].asset_data['trade_date'], self.obj_list[0].asset_data['close'],
                 label=self.obj_list[0].asset_name + '_' + 'close',
                 marker='.', markersize=4,
                 color='k', linestyle='solid', linewidth=1)
        plt.legend(bbox_to_anchor=(1, 1.02), loc='lower right', borderaxespad=0)
        plt.gca().invert_yaxis()

        # plot the strike price
        for i in range(len(self.obj_list)):
            guide_line = float(self.obj_list[i].strike_price)
            plt.axhline(y=guide_line, color='k', linestyle='solid', linewidth=0.5)
            if self.obj_list[i].opt_type == '购':
                guide_line *= 1.0007
            elif self.obj_list[i].opt_type == '沽':
                guide_line *= 0.9993
            plt.axhline(y=guide_line, color='k', linestyle=(0, (5, 5)), linewidth=0.5)

        # set the density of the xtick
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(self.obj_list[0].asset_data) / 20))
        plt.tight_layout()  # adjust figure margin
        plt.show()


class FuzzyQuery(object):
    """Query """

    def __init__(self, asset_type, name):
        self.result_df = self.fuzzy_query(asset_type, name)

    def fuzzy_query(self, asset_type, name):  # noqa
        """Return all the asset containing the queried name.\n
        Parameters:
            asset_type: 'stock', 'index', 'option'
            name: '平安', '上证50', etc
        """
        # query the dataframe which contains information of all the asset
        idx = []
        df = pd.DataFrame()
        if asset_type == 'stock':
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
            idx = df[df['name'].str.contains(name)].index.tolist()
        elif asset_type == 'index':
            df = pro.index_basic(market='', fields='ts_code,name')
            idx = df[df['name'].str.contains(name)].index.tolist()
        elif asset_type == 'ETF':
            df = pro.fund_basic(market='E')
            idx = df[df['name'].str.contains(name)].index.tolist()
        elif asset_type == 'option':
            for opt_exchg in {'CFFEX', 'SSE', 'SZSE', 'SHFE', 'CZCE', 'DCE'}:
                df = pro.opt_basic(exchange=opt_exchg, fields='ts_code,name')
                idx = df[df['name'].str.contains(name)].index.tolist()
                if len(idx) > 0:
                    # break
                    print(df.loc[idx, :])
        if len(idx) == 0:
            print('Asset not found')
            sys.exit()
        else:
            return df.loc[idx, :]  # return a dataframe containing the name

    def df_filter(self, tips=''):
        """Filter by tips, e.g., strike date of option ('2301', etc)"""
        idx = self.result_df[self.result_df['name'].str.contains(tips)].index.tolist()
        return self.result_df.loc[idx, :]

    # noinspection PyMethodMayBeStatic
    def name2optinfo(self, result_df):
        name_df = result_df['name']
        opt_info_list = [dict()]*len(name_df)
        for i in range(len(name_df)):
            # !!! dict must be initialized in the loop, otherwise each appended dict would be the same
            cur_dict = dict(expiration='', opt_type='', strike_price='')
            # get the expiration date
            idx = name_df.iloc[i].find('期权')
            cur_dict['expiration'] = name_df.iloc[i][idx + 2:idx + 6]
            # get the opt_type
            if '购' in name_df.iloc[i]:
                cur_dict['opt_type'] = '购'
                idx = name_df.iloc[i].find('购')
            elif '沽' in name_df.iloc[i]:
                cur_dict['opt_type'] = '沽'
                idx = name_df.iloc[i].find('沽')
            else:
                print('Opttype not found')
                sys.exit()
            # get the strike price
            cur_dict['strike_price'] = name_df.iloc[i][idx + 1:]
            # add to list
            opt_info_list[i] = cur_dict
        return opt_info_list


def align_obj_date(obj_list):
    """fill all the absent dates for the list of class instances,
    where each instance has an attribute *data*"""
    # get all the dates appearing in the stocks
    union_dates = pd.DataFrame(columns=['trade_date'])  # empty dataframe
    for i in range(len(obj_list)):
        union_dates = pd.merge(union_dates, obj_list[i].data['trade_date'], on=['trade_date'], how='outer')

    # fill all the absent dates, and sort
    for i in range(len(obj_list)):
        obj_list[i].data = pd.merge(union_dates, obj_list[i].data, on=['trade_date'], how='outer')
        obj_list[i].data = obj_list[i].data.sort_values(by='trade_date')


def align_date(data_list):
    """fill all the absent dates for the list of dataframe"""

    # get all the dates appearing in the data
    union_dates = pd.DataFrame(columns=['trade_date'])  # empty dataframe with single column name
    # union all the dates, the result is a dataframe with single column ('trade_date')
    for i in range(len(data_list)):
        union_dates = pd.merge(union_dates, data_list[i]['trade_date'], on=['trade_date'], how='outer')

    # fill all the absent dates, and sort
    for i in range(len(data_list)):
        # fill the i-th data
        data_list[i] = pd.merge(union_dates, data_list[i], on=['trade_date'], how='outer')
        data_list[i] = data_list[i].sort_values(by='trade_date')  # sort
    return data_list


def check_name_len(name, max_len):
    temp = list(name)  # to list
    for i in range(math.floor(len(name)/max_len), 0, -1):
        temp.insert(i*max_len, '\n')
    name = ''.join(temp)  # to str
    return name


def plot_obj(obj_list, plot_field=('close',)):
    """plot the trend of a stock/index/...
    plot_field is a *tuple*, which can contain 'close', 'ma5', 'ma10', etc."""

    align_obj_date(obj_list)

    # data_list = []
    # for i in range(len(obj_list)):
    #     data_list.append(obj_list[i].data)
    # data_list = align_date(data_list)

    for i in range(len(obj_list)):
        for j in plot_field:
            plt.plot(obj_list[i].data['trade_date'], obj_list[i].data[j],
                     label=obj_list[i].name+'_'+j, linewidth=1)

    # set the density of the xtick
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(obj_list[0].data)/20))

    plt.xticks(rotation=45)  # set the angle of texts in xtick
    plt.legend(bbox_to_anchor=(0.5, 1.03), loc='lower center', borderaxespad=0)
    plt.show()


def plot_obj_twin(obj_list_1, obj_list_2, plot_field=('close',)):
    """plot the trend of a stock/index/...
    plot_field is a *tuple*, which can contain 'close', 'ma5', 'ma10', etc."""

    align_obj_date(obj_list_1+obj_list_2)

    # data_list = []
    # for i in range(len(obj_list)):
    #     data_list.append(obj_list[i].data)
    # data_list = align_date(data_list)

    for i in range(len(obj_list_1)):
        for j in plot_field:
            plt.plot(obj_list_1[i].data['trade_date'], obj_list_1[i].data[j],
                     label=obj_list_1[i].name+'_'+j, linewidth=1)

    plt.xticks(rotation=45)  # set the angle of texts in xtick
    plt.legend(bbox_to_anchor=(0, 1.02), loc='lower left', borderaxespad=0)

    plt.twinx()
    for i in range(len(obj_list_2)):
        for j in plot_field:
            plt.plot(obj_list_2[i].data['trade_date'], obj_list_2[i].data[j],
                     label=obj_list_2[i].name+'_'+j, linewidth=1)
    plt.legend(bbox_to_anchor=(1, 1.02), loc='lower right', borderaxespad=0)

    # set the density of the xtick
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(obj_list_1[0].data)/20))

    plt.show()


def plot_df(name_list, data_list, plot_field=('close',), plot_type='line'):
    """plot the trend of a stock/index/...
    plot_field is a *tuple*, which can contain 'close', 'ma5', 'ma10', etc."""
    data_list = align_date(data_list)
    for i in range(len(data_list)):
        for j in plot_field:
            if plot_type == 'line':
                plt.plot(data_list[i]['trade_date'], data_list[i][j],
                         label=name_list[i]+'_'+j, linewidth=1)
            elif plot_type == 'bar':
                plt.bar(data_list[i]['trade_date'], data_list[i][j],
                        label=name_list[i] + '_' + j, color='deepskyblue')

    # set the density of the xtick
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(data_list[0])/20))

    plt.xticks(rotation=90)  # set the angle of texts in xtick
    plt.legend(loc='best')
    plt.show()


def end_of_mainbody():
    pass


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    query_date = ('20230101', '20230223')  # (start date, end date)

    '''
    a = FuzzyQuery('ETF', '上证50ETF')
    print(a.result_df)
    # print(a.df_filter('2301'))
    # print(a.name2optinfo(a.df_filter('2301')))
    '''
    '''
    b = Stock('民和股份', query_date)
    c = Stock('圣农发展', query_date)
    # a.moving_average([5])
    print(b.code, b.name)
    comb = Combine([a, b, c], [0.5, 0.5, 0.5])
    comb.show()
    comb.moving_average([5, 10])
    comb.show(['close', 'ma5', 'ma10'])
    # '''
    '''
    # d = Index('上证50', query_date)
    # print(d.name, d.code)
    # d.show()
    # plot_obj_twin([a, b], [comb])
    # plot_obj([c, comb])
    # align_obj_date([a, d])
    # print(a.data)
    '''
    '''
    # a = ETF('上证50ETF')
    a = ETF('沪深300ETF')
    print(a.code, a.name)
    print(a.data)
    '''
    '''
    a = Option('华夏上证50ETF期权2301认购2.90', query_date)
    # print(a.expiration_date)
    print(type(a))
    '''
    # '''
    query_info = dict(asset_name='中证1000指数',
                      asset_tips='',
                      expiration='2303',
                      opt_type='c',
                      strike_price='7400',
                      opt_tips='')
    # query_info = '上证50指数期权2301认沽2750'
    # a = Option(query_info, query_date)
    # print(a.code, a.name, a.data)
    b = OptionAdv(query_info, query_date)
    # print(b.code, b.name)
    print(b.data)
    # print(b.asset.code, b.asset.name)
    # b.show()
    # print(b.asset.data)

    # query_info['opt_type'] = 'c'
    # query_info['strike_price'] = '2800'

    # print(a.asset.code, a.asset.name)
    # print(a.asset.data)
    # '''
    '''
    df = pro.fund_nav(ts_code='510050.SH', start_date=query_date[0], end_date=query_date[1])
    print(df)
    print(a.asset_data)
    '''
