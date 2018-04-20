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
  local should_forward = true
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


      -- HACK caffe does not support the affine translation (scale and shift)
      -- included as part of batch normalization in the original paper. caffe
      -- suggests using a separate Scale layer, which includes a bias shift,
      -- however, there is not a similar layer in torch. this package opts to
      -- translate it to an nn.CMul and nn.Add. however, this makes the next
      -- steps in the process harder. instead, i include two caffe BatchNorm
      -- layers, each containing half the parameters of a torch (true)
      -- nn.BatchNormalization layer and combine them here.

      -- the faux-BatchNormalization layer (first)
      if torch.isTypeOf(module, nn.BatchNormalization) and should_forward then
        -- this node has only a single module and it is a nn.BatchNormalization
        -- layer
        assert(torch.isTypeOf(module, nn.BatchNormalization),
       	    "expected this layer to be of type nn.BatchNormalization")

        -- the next node has only a single module and it is a
        -- nn.BatchNormalization layer
        local nodes1 = modmap[j+1]
        local module1 = nodes1[1].data.module
        module1:float()
        assert(torch.isTypeOf(module1, nn.BatchNormalization),
           "expected the next layer to be of type nn.BatchNormalization")

        -- reach forward and fill out the running mean and variance of the
        -- nn.BatchNormalization layer
        params[(i-1)*2+1] = module1.running_mean and module1.running_mean:cdata() or noData
        params[i*2] = module1.running_var and module1.running_var:cdata() or noData

        should_forward = false
      else
        -- the BatchNormalization layer (second)
        if torch.isTypeOf(module, nn.BatchNormalization) then
          should_forward = true
        end

        params[(i-1)*2+1] = module.weight and module.weight:cdata() or noData
        params[i*2] = module.bias and module.bias:cdata() or noData

        seq:add(module)
      end
    end
    module_params[j] = ffi.new('THFloatTensor*['..#params..']', params)
  end
  local cParams = ffi.new('THFloatTensor**['..#module_params..']', module_params)
  caffegraph.C.getParams(handle, cParams)
  caffegraph.C.freeModel(handle)

  return model, seq
end

return caffegraph

