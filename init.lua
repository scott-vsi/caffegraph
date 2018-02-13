require 'nn'
local ffi = require 'ffi'

caffegraph = {}

ffi.cdef[[
struct params { int num_params; THFloatTensor** params; };
void loadModel(void** handle, const char* prototxt, const char* caffemodel);
void buildModel(void** handle, const char* lua_path);
void getParams(void** handle, THFloatTensor*** params);
void freeModel(void** handle);
]]

caffegraph.C = ffi.load(package.searchpath('libcaffegraph', package.cpath))

caffegraph.load = function(prototxt, caffemodel)
  net, seq = caffegraph.load_both(prototxt, caffemodel)
  return net
end

caffegraph.load_seq = function(prototxt, caffemodel)
  net, seq = caffegraph.load_both(prototxt, caffemodel)
  return seq
end

caffegraph.load_both = function(prototxt, caffemodel)
  local handle = ffi.new('void*[1]')

  -- load the caffemodel into a graph structure
  local initHandle = handle[1]
  caffegraph.C.loadModel(handle, prototxt, caffemodel)
  if handle[1] == initHandle then
    error('Unable to load model.')
  end

  -- serialize the graph and write it out
  local luaModel = path.splitext(caffemodel)
  luaModel = luaModel..'.lua'
  caffegraph.C.buildModel(handle, luaModel)

  -- -- bring the model into lua world
  local model, modmap = dofile(luaModel)
  local seq = nn.Sequential()

  -- transfer the parameters
  local noData = torch.FloatTensor():zero():cdata()
  local module_params = {}
  for j,nodes in ipairs(modmap) do
    local params = {}
    for i=1,#nodes do
      local module = nodes[i].data.module
      module:float()

      --if torch.isTypeOf(module, nn.BatchNormalization) then
      --  module.weight:fill(1)
      --  module.bias:zero()
      --  params[(i-1)*2+1] = module.running_mean:cdata()
      --  params[i*2] = module.running_var:cdata()
      --else
      --  params[(i-1)*2+1] = module.weight and module.weight:cdata() or noData
      --  params[i*2] = module.bias and module.bias:cdata() or noData
      --end

      if torch.isTypeOf(module, nn.BatchNormalization) then
        -- HACK in dlib, a batch norm layer is swapped out at test time for an
        -- affine layer (a scale and shift). there is not an affine layer in
        -- torch, and this package ops to translate it to an nn.CMul and nn.Add.
        -- however, this makes the next steps in the process harder. instead,
        -- i hijack the affine transformation included (by definition) in the
        -- nn.BatchNormalization layer (used while traning) to perform an
        -- affine transformation
        module.running_mean:zero()
        module.running_var:fill(1)
      end
      params[(i-1)*2+1] = module.weight and module.weight:cdata() or noData
      params[i*2] = module.bias and module.bias:cdata() or noData

      seq:add(module)
    end
    module_params[j] = ffi.new('THFloatTensor*['..#params..']', params)
  end
  local cParams = ffi.new('THFloatTensor**['..#module_params..']', module_params)
  caffegraph.C.getParams(handle, cParams)
  caffegraph.C.freeModel(handle)

  return model, seq
end

return caffegraph
